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
}
