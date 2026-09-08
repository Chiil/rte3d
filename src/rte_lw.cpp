#include "rte_lw.h"
#include "rte_solver_kernels.h"

using Rte_kernels::Vert;
using Rte_kernels::pi;
using Rte_kernels::adding_column;


// The reference's lw_solver_noscat also offers do_rescaling, the approximate treatment
// of scattering of Tang et al. 2018 (10.1175/JAS-D-18-0014.1), through
// lw_transport_1rescl. That is not implemented here. Its two orientation branches read
// the upward radiance at different levels relative to the layer during the second
// downward sweep -- the top-at-1 branch at the upstream level, the other at the
// destination level -- and that asymmetry should be resolved against the reference
// authors before it is reproduced.
namespace
{
    template<bool top_at_1>
    void solver_noscat_impl(
            const Array_3d<const TF>& secants,
            const Array_1d<const TF>& weights,
            const Array_3d<const TF>& tau,
            const Source_func_lw& sources,
            const Array_2d<const TF>& sfc_emis,
            const Array_2d<const TF>& inc_flux,
            const Array_3d<TF>& flux_up,
            const Array_3d<TF>& flux_dn,
            const Array_2d<TF>& flux_up_jac)
    {
        using V = Vert<top_at_1>;

        const int ngpt = static_cast<int>(tau.extent(0));
        const int nlay = static_cast<int>(tau.extent(1));
        const int ncol = static_cast<int>(tau.extent(2));
        const int nlev = nlay + 1;
        const int nmus = static_cast<int>(weights.extent(0));

        const bool do_jacobians = flux_up_jac.size() > 0;

        const Array_3d<const TF> lay_source = sources.lay_source;
        const Array_3d<const TF> lev_source = sources.lev_source;
        const Array_2d<const TF> sfc_source = sources.sfc_source;
        const Array_2d<const TF> sfc_source_jac = sources.sfc_source_jac;

        // Per-column scratch. The reference keeps these (ncol, nlay) because it loops
        // g-points serially; running them in parallel adds the g-point dimension. See
        // the note in rte_sw.cpp about blocking this later.
        Array_3d<TF> trans(Kokkos::view_alloc("trans", Kokkos::WithoutInitializing), ngpt, nlay, ncol);
        Array_3d<TF> source_up(Kokkos::view_alloc("source_up", Kokkos::WithoutInitializing), ngpt, nlay, ncol);
        Array_3d<TF> source_dn(Kokkos::view_alloc("source_dn", Kokkos::WithoutInitializing), ngpt, nlay, ncol);
        Array_3d<TF> rad_up(Kokkos::view_alloc("rad_up", Kokkos::WithoutInitializing), ngpt, nlev, ncol);
        Array_3d<TF> rad_dn(Kokkos::view_alloc("rad_dn", Kokkos::WithoutInitializing), ngpt, nlev, ncol);

        // The Jacobian is spectrally integrated, so it is accumulated per g-point here
        // and reduced afterwards rather than written to with atomics.
        Array_3d<TF> rad_up_jac("rad_up_jac", do_jacobians ? ngpt : 0, do_jacobians ? nlev : 0,
                                do_jacobians ? ncol : 0);

        Kokkos::deep_copy(flux_up, TF(0.));
        Kokkos::deep_copy(flux_dn, TF(0.));
        if (do_jacobians)
            Kokkos::deep_copy(flux_up_jac, TF(0.));

        // The weights are needed on the host to scale each angle's contribution.
        const auto weights_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weights);

        // Each quadrature angle is an independent single-angle solve; the reference
        // sums them, and so do we.
        for (int imu=0; imu<nmus; ++imu)
        {
            const TF scaling = pi * weights_h(imu);

            parallel_for_gpt_col("lw_solver_noscat", ngpt, ncol,
                KOKKOS_LAMBDA(const int igpt, const int icol)
                {
                    const TF D = secants(imu, igpt, icol);

                    // Transport is for intensity: convert the flux at the top of the
                    // domain to intensity assuming azimuthal isotropy.
                    rad_dn(igpt, V::lev_toa(nlay), icol) = inc_flux(igpt, icol) / scaling;

                    // Optical path, transmission and the layer source functions.
                    for (int ilay=0; ilay<nlay; ++ilay)
                    {
                        const TF tau_loc = tau(igpt, ilay, icol) * D;
                        const TF trans_l = Kokkos::exp(-tau_loc);
                        trans(igpt, ilay, icol) = trans_l;

                        TF source_up_l, source_dn_l;
                        Rte_kernels::lw_source_noscat(
                                lay_source(igpt, ilay, icol),
                                lev_source(igpt, ilay + V::lev_up(), icol),
                                lev_source(igpt, ilay + V::lev_dn(), icol),
                                tau_loc, trans_l, source_up_l, source_dn_l);

                        source_up(igpt, ilay, icol) = source_up_l;
                        source_dn(igpt, ilay, icol) = source_dn_l;
                    }

                    // Transport down, from the top of the atmosphere.
                    for (int j=0; j<nlay; ++j)
                    {
                        const int ilay = V::lay_from_toa(j, nlay);
                        rad_dn(igpt, ilay + V::lev_dn(), icol) =
                                trans(igpt, ilay, icol) * rad_dn(igpt, ilay + V::lev_up(), icol)
                                + source_dn(igpt, ilay, icol);
                    }

                    // Surface reflection and emission.
                    const int lev_sfc = V::lev_sfc(nlay);
                    const TF emis = sfc_emis(igpt, icol);

                    rad_up(igpt, lev_sfc, icol) = rad_dn(igpt, lev_sfc, icol) * (TF(1.) - emis)
                                                + emis * sfc_source(igpt, icol);
                    if (do_jacobians)
                        rad_up_jac(igpt, lev_sfc, icol) = emis * sfc_source_jac(igpt, icol);

                    // Transport up, from the surface.
                    for (int j=0; j<nlay; ++j)
                    {
                        const int ilay = V::lay_from_sfc(j, nlay);
                        const TF trans_l = trans(igpt, ilay, icol);

                        rad_up(igpt, ilay + V::lev_up(), icol) =
                                trans_l * rad_up(igpt, ilay + V::lev_dn(), icol)
                                + source_up(igpt, ilay, icol);

                        if (do_jacobians)
                            rad_up_jac(igpt, ilay + V::lev_up(), icol) =
                                    trans_l * rad_up_jac(igpt, ilay + V::lev_dn(), icol);
                    }

                    // Convert intensity back to flux, assuming azimuthal isotropy, and
                    // accumulate this angle's contribution.
                    for (int ilev=0; ilev<nlev; ++ilev)
                    {
                        flux_up(igpt, ilev, icol) += scaling * rad_up(igpt, ilev, icol);
                        flux_dn(igpt, ilev, icol) += scaling * rad_dn(igpt, ilev, icol);
                    }
                });

            // Only broadband Jacobians are provided, so reduce over g-points.
            if (do_jacobians)
            {
                parallel_for_2d("lw_solver_noscat_jacobian", {0, 0}, {nlev, ncol},
                    KOKKOS_LAMBDA(const int ilev, const int icol)
                    {
                        TF sum = TF(0.);
                        for (int igpt=0; igpt<ngpt; ++igpt)
                            sum += rad_up_jac(igpt, ilev, icol);

                        flux_up_jac(ilev, icol) += scaling * sum;
                    });
            }
        }
    }
}


void Rte_lw::solver_noscat(
        const bool top_at_1,
        const Array_3d<const TF>& secants,
        const Array_1d<const TF>& weights,
        const Array_3d<const TF>& tau,
        const Source_func_lw& sources,
        const Array_2d<const TF>& sfc_emis,
        const Array_2d<const TF>& inc_flux,
        const Array_3d<TF>& flux_up,
        const Array_3d<TF>& flux_dn,
        const Array_2d<TF>& flux_up_jac)
{
    if (top_at_1)
        solver_noscat_impl<true>(
                secants, weights, tau, sources, sfc_emis, inc_flux, flux_up, flux_dn, flux_up_jac);
    else
        solver_noscat_impl<false>(
                secants, weights, tau, sources, sfc_emis, inc_flux, flux_up, flux_dn, flux_up_jac);
}


namespace
{
    template<bool top_at_1>
    void solver_2stream_impl(
            const Array_3d<const TF>& tau,
            const Array_3d<const TF>& ssa,
            const Array_3d<const TF>& g,
            const Source_func_lw& sources,
            const Array_2d<const TF>& sfc_emis,
            const Array_2d<const TF>& inc_flux,
            const Array_3d<TF>& flux_up,
            const Array_3d<TF>& flux_dn)
    {
        using V = Vert<top_at_1>;

        const int ngpt = static_cast<int>(tau.extent(0));
        const int nlay = static_cast<int>(tau.extent(1));
        const int ncol = static_cast<int>(tau.extent(2));
        const int nlev = nlay + 1;

        const Array_3d<const TF> lev_source = sources.lev_source;
        const Array_2d<const TF> sfc_source = sources.sfc_source;

        Array_3d<TF> Rdif(Kokkos::view_alloc("Rdif", Kokkos::WithoutInitializing), ngpt, nlay, ncol);
        Array_3d<TF> Tdif(Kokkos::view_alloc("Tdif", Kokkos::WithoutInitializing), ngpt, nlay, ncol);
        Array_3d<TF> source_up(Kokkos::view_alloc("source_up", Kokkos::WithoutInitializing), ngpt, nlay, ncol);
        Array_3d<TF> source_dn(Kokkos::view_alloc("source_dn", Kokkos::WithoutInitializing), ngpt, nlay, ncol);
        Array_3d<TF> albedo(Kokkos::view_alloc("albedo", Kokkos::WithoutInitializing), ngpt, nlev, ncol);
        Array_3d<TF> src(Kokkos::view_alloc("src", Kokkos::WithoutInitializing), ngpt, nlev, ncol);
        Array_3d<TF> denom(Kokkos::view_alloc("denom", Kokkos::WithoutInitializing), ngpt, nlay, ncol);

        const Array_3d<const TF> Rdif_c = Rdif;
        const Array_3d<const TF> Tdif_c = Tdif;
        const Array_3d<const TF> source_up_c = source_up;
        const Array_3d<const TF> source_dn_c = source_dn;

        parallel_for_gpt_col("lw_solver_2stream", ngpt, ncol,
            KOKKOS_LAMBDA(const int igpt, const int icol)
            {
                // Cell properties, and the source function for diffuse radiation.
                for (int ilay=0; ilay<nlay; ++ilay)
                {
                    const TF tau_l = tau(igpt, ilay, icol);

                    TF gamma1, gamma2, Rdif_l, Tdif_l;
                    Rte_kernels::lw_two_stream(
                            tau_l, ssa(igpt, ilay, icol), g(igpt, ilay, icol),
                            gamma1, gamma2, Rdif_l, Tdif_l);

                    Rdif(igpt, ilay, icol) = Rdif_l;
                    Tdif(igpt, ilay, icol) = Tdif_l;

                    TF source_up_l, source_dn_l;
                    Rte_kernels::lw_source_2str(
                            lev_source(igpt, ilay + V::lev_up(), icol),
                            lev_source(igpt, ilay + V::lev_dn(), icol),
                            gamma1, gamma2, Rdif_l, Tdif_l, tau_l,
                            source_up_l, source_dn_l);

                    source_up(igpt, ilay, icol) = source_up_l;
                    source_dn(igpt, ilay, icol) = source_dn_l;
                }

                const TF emis = sfc_emis(igpt, icol);
                const TF source_sfc = pi * emis * sfc_source(igpt, icol);

                // Boundary condition on the diffuse downward flux.
                flux_dn(igpt, V::lev_toa(nlay), icol) = inc_flux(igpt, icol);

                adding_column<top_at_1>(
                        igpt, icol, nlay,
                        TF(1.) - emis,
                        Rdif_c, Tdif_c, source_dn_c, source_up_c, source_sfc,
                        flux_up, flux_dn,
                        albedo, src, denom);
            });
    }
}


void Rte_lw::solver_2stream(
        const bool top_at_1,
        const Array_3d<const TF>& tau,
        const Array_3d<const TF>& ssa,
        const Array_3d<const TF>& g,
        const Source_func_lw& sources,
        const Array_2d<const TF>& sfc_emis,
        const Array_2d<const TF>& inc_flux,
        const Array_3d<TF>& flux_up,
        const Array_3d<TF>& flux_dn)
{
    if (top_at_1)
        solver_2stream_impl<true>(tau, ssa, g, sources, sfc_emis, inc_flux, flux_up, flux_dn);
    else
        solver_2stream_impl<false>(tau, ssa, g, sources, sfc_emis, inc_flux, flux_up, flux_dn);
}
