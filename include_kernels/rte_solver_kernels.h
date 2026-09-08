#pragma once

#include <limits>

#include "types.h"


namespace Rte_kernels
{
    inline constexpr TF eps = std::numeric_limits<TF>::epsilon();

    // Lower limit on k, suggested by Chiel van Heerwaarden: k = 0 for isotropic,
    // conservative scattering, and this limit keeps the relative error in Rdif below
    // 0.1% down to tau = 1e-9 while avoiding the division by zero.
    inline constexpr TF min_k = TF(1.e4) * eps;


    // Vertical orientation of the arrays. The reference writes every loop out twice,
    // once per orientation; instead we carry the two level offsets of a layer and
    // template the solvers on the orientation, so the compiler still sees explicit
    // loops but the physics is written once.
    //
    // Layer ilay is bounded by levels ilay+lev_up() and ilay+lev_dn(), where "up" and
    // "dn" are directions in space, not array order.
    template<bool top_at_1>
    struct Vert
    {
        KOKKOS_INLINE_FUNCTION static constexpr int lev_up() { return top_at_1 ? 0 : 1; }
        KOKKOS_INLINE_FUNCTION static constexpr int lev_dn() { return top_at_1 ? 1 : 0; }

        KOKKOS_INLINE_FUNCTION static constexpr int lev_toa(const int nlay) { return top_at_1 ? 0 : nlay; }
        KOKKOS_INLINE_FUNCTION static constexpr int lev_sfc(const int nlay) { return top_at_1 ? nlay : 0; }

        // Layer visited as the j-th step of a sweep that starts at the top of the
        // atmosphere and works downward.
        KOKKOS_INLINE_FUNCTION static constexpr int lay_from_toa(const int j, const int nlay)
        { return top_at_1 ? j : nlay-1-j; }

        // Layer visited as the j-th step of a sweep that starts at the surface and
        // works upward.
        KOKKOS_INLINE_FUNCTION static constexpr int lay_from_sfc(const int j, const int nlay)
        { return top_at_1 ? nlay-1-j : j; }
    };


    // Zdunkowski Practical Improved Flux Method "PIFM" (Zdunkowski et al., 1980,
    // Contributions to Atmospheric Physics 53, 147-66), with the Meador and Weaver
    // direct-beam terms. Reference: sw_dif_and_source in mo_rte_solver_kernels.F90.
    //
    // Returns the layer's diffuse reflectance and transmittance, the direct-beam
    // reflectance and transmittance into the diffuse field, and the unscattered
    // direct transmittance.
    KOKKOS_INLINE_FUNCTION
    void sw_two_stream(
            const TF tau, const TF w0, const TF g, const TF mu0,
            TF& Rdif, TF& Tdif, TF& Rdir, TF& Tdir, TF& Tnoscat)
    {
        const TF gamma1 = (TF(8.) - w0 * (TF(5.) + TF(3.)*g)) * TF(0.25);
        const TF gamma2 = (TF(3.) * (w0 * (TF(1.) - g))) * TF(0.25);

        // Eq 18; k = sqrt(gamma1^2 - gamma2^2), limited below to avoid dividing by 0.
        const TF k = Kokkos::sqrt(Kokkos::max((gamma1 - gamma2) * (gamma1 + gamma2), min_k));
        const TF exp_minusktau = Kokkos::exp(-tau*k);
        const TF exp_minus2ktau = exp_minusktau * exp_minusktau;

        // Refactored to avoid rounding errors when k and gamma1 differ greatly in magnitude.
        TF RT_term = TF(1.) / (k      * (TF(1.) + exp_minus2ktau) +
                               gamma1 * (TF(1.) - exp_minus2ktau));

        Rdif = RT_term * gamma2 * (TF(1.) - exp_minus2ktau);  // Eq 25
        Tdif = RT_term * TF(2.) * k * exp_minusktau;          // Eq 26

        // On a round earth mu0 can increase with depth, so levels with mu0 <= 0 have no
        // direct beam. Compute with a nominal value here and mask the result at the
        // call site.
        const TF mu0_s = Kokkos::max(Kokkos::sqrt(eps), mu0);
        const TF k_mu = k * mu0_s;

        // Eq 14, top and bottom multiplied by exp(-k*tau) and rearranged to avoid a
        // division by zero.
        const TF one_minus_kmu2 = TF(1.) - k_mu*k_mu;
        RT_term = w0 * RT_term / (Kokkos::abs(one_minus_kmu2) >= eps ? one_minus_kmu2 : eps);

        const TF gamma3 = (TF(2.) - TF(3.) * mu0_s * g) * TF(0.25);
        const TF gamma4 = TF(1.) - gamma3;
        const TF alpha1 = gamma1 * gamma4 + gamma2 * gamma3;  // Eq 16
        const TF alpha2 = gamma1 * gamma3 + gamma2 * gamma4;  // Eq 17

        const TF k_gamma3 = k * gamma3;
        const TF k_gamma4 = k * gamma4;

        Tnoscat = Kokkos::exp(-tau/mu0_s);

        Rdir = RT_term *
            ((TF(1.) - k_mu) * (alpha2 + k_gamma3)                  -
             (TF(1.) + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau -
             TF(2.) * (k_gamma3 - alpha2 * k_mu) * exp_minusktau * Tnoscat);

        // Eq 15, top and bottom multiplied by exp(-k*tau) and the whole multiplied
        // through by exp(-tau/mu0) to prefer underflow to overflow. The direct
        // transmittance is omitted.
        Tdir = -RT_term *
            ((TF(1.) + k_mu) * (alpha1 + k_gamma4)                  * Tnoscat -
             (TF(1.) - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * Tnoscat -
             TF(2.) * (k_gamma4 + alpha1 * k_mu) * exp_minusktau);

        // The beam is reflected, penetrates unscattered, or penetrates and is
        // scattered on the way; the rest is absorbed. Clamping to that budget keeps
        // the equations safe in single precision. Credit: Robin Hogan, Peter Ukkonen.
        Rdir = Kokkos::max(TF(0.), Kokkos::min(Rdir, TF(1.) - Tnoscat));
        Tdir = Kokkos::max(TF(0.), Kokkos::min(Tdir, TF(1.) - Tnoscat - Rdir));
    }

    inline constexpr TF pi = TF(3.14159265358979323846);

    // Longwave source function for diffuse radiation, using the linear-in-tau
    // assumption of Clough et al., 1992, doi:10.1029/92JD01419, Eq 13.
    //
    // lev_source_up and lev_source_dn are the Planck sources at the level on the
    // upward and downward side of the layer, which is what the reference selects with
    // its source_inc / source_dec pointer swap.
    KOKKOS_INLINE_FUNCTION
    void lw_source_noscat(
            const TF lay_source, const TF lev_source_up, const TF lev_source_dn,
            const TF tau_loc, const TF trans,
            TF& source_up, TF& source_dn)
    {
        // Weighting factor. Below the threshold the rounding error in the direct form
        // (~tau^2) is of order epsilon, so use a 3rd order series expansion instead.
        // Thanks to Peter Blossey (UW) for the idea and Dmitry Alexeev (Nvidia) for
        // suggesting 3rd order.
        const TF tau_thresh = Kokkos::sqrt(Kokkos::sqrt(eps));

        const TF fact = tau_loc > tau_thresh
                ? (TF(1.) - trans)/tau_loc - trans
                : tau_loc * (TF(0.5) + tau_loc * (TF(-1.)/TF(3.) + tau_loc * TF(1.)/TF(8.)));

        source_dn = (TF(1.) - trans) * lev_source_dn + TF(2.) * fact * (lay_source - lev_source_dn);
        source_up = (TF(1.) - trans) * lev_source_up + TF(2.) * fact * (lay_source - lev_source_up);
    }

    // Transport of diffuse radiation through a vertically layered atmosphere, after
    // Shonk and Hogan 2008, doi:10.1175/2007JCLI1940.1 (SH08). Shared by longwave and
    // shortwave. Reference: adding in mo_rte_solver_kernels.F90.
    //
    // Runs entirely inside one (igpt, icol) thread: two sequential sweeps over the
    // column, so albedo, src and denom are per-column scratch.
    template<bool top_at_1, typename Albedo, typename Src, typename Denom>
    KOKKOS_INLINE_FUNCTION
    void adding_column(
            const int igpt, const int icol, const int nlay,
            const TF albedo_sfc,
            const Array_3d<const TF>& rdif, const Array_3d<const TF>& tdif,
            const Array_3d<const TF>& src_dn, const Array_3d<const TF>& src_up,
            const TF src_sfc,
            const Array_3d<TF>& flux_up, const Array_3d<TF>& flux_dn,
            const Albedo& albedo, const Src& src, const Denom& denom)
    {
        using V = Vert<top_at_1>;

        // Reflectivity to diffuse radiation below this level (alpha in SH08) and the
        // source of diffuse upwelling radiation (G in SH08) start at the surface.
        albedo(igpt, V::lev_sfc(nlay), icol) = albedo_sfc;
        src   (igpt, V::lev_sfc(nlay), icol) = src_sfc;

        // From the surface upward, accumulate albedo and the source of upward radiation.
        for (int j=0; j<nlay; ++j)
        {
            const int ilay = V::lay_from_sfc(j, nlay);
            const int lev_above = ilay + V::lev_up();
            const int lev_below = ilay + V::lev_dn();

            const TF rdif_l = rdif(igpt, ilay, icol);
            const TF tdif_l = tdif(igpt, ilay, icol);

            const TF denom_l = TF(1.) / (TF(1.) - rdif_l * albedo(igpt, lev_below, icol));  // Eq 10
            denom(igpt, ilay, icol) = denom_l;

            albedo(igpt, lev_above, icol) =
                    rdif_l + tdif_l*tdif_l * albedo(igpt, lev_below, icol) * denom_l;  // Eq 9

            // Eq 11: upward emission at the top of the layer, plus radiation emitted at
            // the bottom, transmitted through and reflected from the layers below.
            src(igpt, lev_above, icol) =
                    src_up(igpt, ilay, icol)
                    + tdif_l * denom_l * (src(igpt, lev_below, icol)
                                          + albedo(igpt, lev_below, icol) * src_dn(igpt, ilay, icol));
        }

        // Eq 12 at the top of the domain: reflection of the incident diffuse flux plus
        // emission from below.
        {
            const int lev = V::lev_toa(nlay);
            flux_up(igpt, lev, icol) = flux_dn(igpt, lev, icol) * albedo(igpt, lev, icol)
                                     + src(igpt, lev, icol);
        }

        // From the top of the atmosphere downward, compute the fluxes.
        for (int j=0; j<nlay; ++j)
        {
            const int ilay = V::lay_from_toa(j, nlay);
            const int lev_prev = ilay + V::lev_up();
            const int lev_dst  = ilay + V::lev_dn();

            flux_dn(igpt, lev_dst, icol) =                                       // Eq 13
                    (tdif(igpt, ilay, icol) * flux_dn(igpt, lev_prev, icol)
                     + rdif(igpt, ilay, icol) * src(igpt, lev_dst, icol)
                     + src_dn(igpt, ilay, icol)) * denom(igpt, ilay, icol);

            flux_up(igpt, lev_dst, icol) =                                       // Eq 12
                    flux_dn(igpt, lev_dst, icol) * albedo(igpt, lev_dst, icol)
                    + src(igpt, lev_dst, icol);
        }
    }


    // Longwave two-stream diffuse reflectance and transmittance for a layer, plus the
    // coupling coefficients the source function needs. Reference: lw_two_stream.
    //
    // The coefficients differ from the shortwave because the phase function is more
    // isotropic; we follow Fu et al. 1997,
    // doi:10.1175/1520-0469(1997)054<2799:MSPITI>2.0.CO;2, with a diffusivity secant
    // of 1.66.
    KOKKOS_INLINE_FUNCTION
    void lw_two_stream(
            const TF tau, const TF w0, const TF g,
            TF& gamma1, TF& gamma2, TF& Rdif, TF& Tdif)
    {
        // The reference writes this as `real(wp), parameter :: LW_diff_sec = 1.66`.
        // The literal has default (single) real kind and is only then promoted, so its
        // value is 1.65999996662139893, not 1.66 -- a 2e-8 relative difference. Match
        // it exactly rather than "fix" it, or every comparison drifts by that much.
        constexpr TF lw_diff_sec = static_cast<TF>(1.66f);

        gamma1 = lw_diff_sec * (TF(1.) - TF(0.5) * w0 * (TF(1.) + g));  // Fu et al. Eq 2.9
        gamma2 = lw_diff_sec *           TF(0.5) * w0 * (TF(1.) - g);   // Fu et al. Eq 2.10

        // Eq 18; k = sqrt(gamma1^2 - gamma2^2). Note the floor is a plain 1e-12 here,
        // not the epsilon-derived min_k the shortwave uses.
        const TF k = Kokkos::sqrt(Kokkos::max((gamma1 - gamma2) * (gamma1 + gamma2), TF(1.e-12)));

        const TF exp_minusktau = Kokkos::exp(-tau*k);
        const TF exp_minus2ktau = exp_minusktau * exp_minusktau;

        // Refactored to avoid rounding errors when k and gamma1 differ greatly in magnitude.
        const TF RT_term = TF(1.) / (k      * (TF(1.) + exp_minus2ktau) +
                                     gamma1 * (TF(1.) - exp_minus2ktau));

        Rdif = RT_term * gamma2 * (TF(1.) - exp_minus2ktau);  // Eq 25
        Tdif = RT_term * TF(2.) * k * exp_minusktau;          // Eq 26
    }


    // Longwave source function for upward and downward emission at levels, using the
    // linear-in-tau assumption. Straight from ECRAD, via Toon et al. (JGR 1989)
    // Eqs 26-27. Sources come out in flux units, hence the factor of pi.
    //
    // lev_source_top and lev_source_bot are the Planck sources at the levels above and
    // below the layer: ilay + lev_up() and ilay + lev_dn().
    KOKKOS_INLINE_FUNCTION
    void lw_source_2str(
            const TF lev_source_top, const TF lev_source_bot,
            const TF gamma1, const TF gamma2, const TF rdif, const TF tdif, const TF tau,
            TF& source_up, TF& source_dn)
    {
        if (tau > TF(1.0e-8))
        {
            const TF Z = (lev_source_bot - lev_source_top) / (tau * (gamma1 + gamma2));

            const TF Zup_top    =  Z + lev_source_top;
            const TF Zup_bottom =  Z + lev_source_bot;
            const TF Zdn_top    = -Z + lev_source_top;
            const TF Zdn_bottom = -Z + lev_source_bot;

            source_up = pi * (Zup_top    - rdif * Zdn_top    - tdif * Zup_bottom);
            source_dn = pi * (Zdn_bottom - rdif * Zup_bottom - tdif * Zdn_top);
        }
        else
        {
            source_up = TF(0.);
            source_dn = TF(0.);
        }
    }
}
