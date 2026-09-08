#include "rte_sw.h"
#include "rte_solver_kernels.h"

using Rte_kernels::Vert;
using Rte_kernels::adding_column;


namespace
{
    template<bool top_at_1>
    void solver_noscat_impl(
            const Array_3d<const TF>& tau,
            const Array_2d<const TF>& mu0,
            const Array_2d<const TF>& inc_flux_dir,
            const Array_3d<TF>& flux_dir)
    {
        using V = Vert<top_at_1>;

        const int ngpt = static_cast<int>(tau.extent(0));
        const int nlay = static_cast<int>(tau.extent(1));
        const int ncol = static_cast<int>(tau.extent(2));

        parallel_for_gpt_col("sw_solver_noscat", ngpt, ncol,
            KOKKOS_LAMBDA(const int igpt, const int icol)
            {
                const int lay_toa = V::lay_from_toa(0, nlay);
                flux_dir(igpt, V::lev_toa(nlay), icol) = inc_flux_dir(igpt, icol) * mu0(lay_toa, icol);

                for (int j=0; j<nlay; ++j)
                {
                    const int ilay = V::lay_from_toa(j, nlay);
                    flux_dir(igpt, ilay + V::lev_dn(), icol) =
                            flux_dir(igpt, ilay + V::lev_up(), icol)
                            * Kokkos::exp(-tau(igpt, ilay, icol) / mu0(ilay, icol));
                }
            });
    }


}


namespace
{
    template<bool top_at_1>
    void solver_2stream_impl(
            const Array_3d<const TF>& tau,
            const Array_3d<const TF>& ssa,
            const Array_3d<const TF>& g,
            const Array_2d<const TF>& mu0,
            const Array_2d<const TF>& sfc_alb_dir,
            const Array_2d<const TF>& sfc_alb_dif,
            const Array_2d<const TF>& inc_flux_dir,
            const Array_2d<const TF>& inc_flux_dif,
            const Array_3d<TF>& flux_up,
            const Array_3d<TF>& flux_dn,
            const Array_3d<TF>& flux_dir)
    {
        using V = Vert<top_at_1>;

        const int ngpt = static_cast<int>(tau.extent(0));
        const int nlay = static_cast<int>(tau.extent(1));
        const int ncol = static_cast<int>(tau.extent(2));
        const int nlev = nlay + 1;

        const bool has_dif_bc = inc_flux_dif.size() > 0;

        // The reference loops g-points serially and keeps (ncol, nlay) scratch. We run
        // them in parallel, so the scratch carries a g-point dimension. That is what
        // buys a single CPU/GPU code path; the cost is ~7 arrays of (ngpt, nlay, ncol),
        // which for large column counts will want g-point blocking later.
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

        parallel_for_gpt_col("sw_solver_2stream", ngpt, ncol,
            KOKKOS_LAMBDA(const int igpt, const int icol)
            {
                const int lev_toa = V::lev_toa(nlay);

                // Boundary conditions: the direct beam, and the diffuse field using
                // zero when no condition is supplied.
                flux_dir(igpt, lev_toa, icol) =
                        inc_flux_dir(igpt, icol) * mu0(V::lay_from_toa(0, nlay), icol);
                flux_dn(igpt, lev_toa, icol) = has_dif_bc ? inc_flux_dif(igpt, icol) : TF(0.);

                // Cell properties, and the direct beam attenuating downward.
                for (int j=0; j<nlay; ++j)
                {
                    const int ilay = V::lay_from_toa(j, nlay);
                    const TF mu0_l = mu0(ilay, icol);

                    TF Rdif_l, Tdif_l, Rdir_l, Tdir_l, Tnoscat_l;
                    Rte_kernels::sw_two_stream(
                            tau(igpt, ilay, icol), ssa(igpt, ilay, icol), g(igpt, ilay, icol), mu0_l,
                            Rdif_l, Tdif_l, Rdir_l, Tdir_l, Tnoscat_l);

                    Rdif(igpt, ilay, icol) = Rdif_l;
                    Tdif(igpt, ilay, icol) = Tdif_l;

                    const TF dir_inc = flux_dir(igpt, ilay + V::lev_up(), icol);

                    // T and R for the direct beam were computed with a nominal mu0 even
                    // where the sun is below the horizon; zero those out again.
                    const bool sunlit = mu0_l > TF(0.);
                    source_up(igpt, ilay, icol) = sunlit ? Rdir_l * dir_inc : TF(0.);
                    source_dn(igpt, ilay, icol) = sunlit ? Tdir_l * dir_inc : TF(0.);

                    flux_dir(igpt, ilay + V::lev_dn(), icol) = Tnoscat_l * dir_inc;
                }

                // Source for upward radiation at the surface.
                const int lay_sfc = V::lay_from_sfc(0, nlay);
                const TF source_sfc = mu0(lay_sfc, icol) > TF(0.)
                        ? flux_dir(igpt, V::lev_sfc(nlay), icol) * sfc_alb_dir(igpt, icol)
                        : TF(0.);

                adding_column<top_at_1>(
                        igpt, icol, nlay,
                        sfc_alb_dif(igpt, icol),
                        Rdif_c, Tdif_c, source_dn_c, source_up_c, source_sfc,
                        flux_up, flux_dn,
                        albedo, src, denom);

                // adding() computes only the diffuse flux; flux_dn is the total.
                for (int ilev=0; ilev<nlev; ++ilev)
                    flux_dn(igpt, ilev, icol) += flux_dir(igpt, ilev, icol);
            });
    }
}


void Rte_sw::solver_noscat(
        const bool top_at_1,
        const Array_3d<const TF>& tau,
        const Array_2d<const TF>& mu0,
        const Array_2d<const TF>& inc_flux_dir,
        const Array_3d<TF>& flux_dir)
{
    if (top_at_1)
        solver_noscat_impl<true>(tau, mu0, inc_flux_dir, flux_dir);
    else
        solver_noscat_impl<false>(tau, mu0, inc_flux_dir, flux_dir);
}


void Rte_sw::solver_2stream(
        const bool top_at_1,
        const Array_3d<const TF>& tau,
        const Array_3d<const TF>& ssa,
        const Array_3d<const TF>& g,
        const Array_2d<const TF>& mu0,
        const Array_2d<const TF>& sfc_alb_dir,
        const Array_2d<const TF>& sfc_alb_dif,
        const Array_2d<const TF>& inc_flux_dir,
        const Array_2d<const TF>& inc_flux_dif,
        const Array_3d<TF>& flux_up,
        const Array_3d<TF>& flux_dn,
        const Array_3d<TF>& flux_dir)
{
    if (top_at_1)
        solver_2stream_impl<true>(
                tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir, inc_flux_dif,
                flux_up, flux_dn, flux_dir);
    else
        solver_2stream_impl<false>(
                tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir, inc_flux_dif,
                flux_up, flux_dn, flux_dir);
}
