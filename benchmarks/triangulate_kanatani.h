#pragma once
/*
    Kanatani Optimal Correction Triangulation
    
    Based on: "Triangulation from Two Views Revisited: Hartley-Sturm vs. 
               Optimal Correction" by Kanatani, Sugaya, and Niitsuma (2008)
    
    This implementation provides two variants:
    1. kanatani_optimal    : Full optimal correction (slow but thorough)
    2. kanatani_simplified : Simplified iterative method (recommended)
    
    COORDINATE CONVENTION:
    For fundamental matrix F and points x ↔ x', the epipolar constraint is:
        x'ᵀ F x = 0
    
    Where:
    - x is in the FIRST camera
    - x' is in the SECOND camera
    
    CONVERGENCE:
    - Kanatani's method converges to a local extremum
    - Satisfies epipolar constraint upon convergence
    - May require 2-10 iterations depending on configuration
    - Unstable camera configurations may require more iterations
    
    PERFORMANCE:
    - Faster than Hartley-Sturm polynomial method
    - Slower than Lindström's quadratic method
    - More reliable than first-order approximations
*/

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <limits>

namespace triangulation {

//=============================================================================
// Kanatani Simplified Method (Recommended)
//=============================================================================

/**
 * @brief Kanatani simplified optimal correction
 * 
 * This is the simplified version presented in Lindström's paper (Listing 1).
 * It's much faster than the full method and works very well in practice.
 * 
 * Algorithm:
 * 1. Start with measured points as initial estimate
 * 2. Compute normals to current epipolar lines
 * 3. Solve linear equation for step size λ
 * 4. Update point estimates
 * 5. Repeat until convergence
 * 
 * @param x Points in FIRST camera, shape (3, N), homogeneous
 * @param xp Points in SECOND camera, shape (3, N), homogeneous
 * @param F Fundamental matrix such that xp'Fx = 0
 * @param A Output: corrected points [xp_corrected; x_corrected], shape (4, N)
 * @param max_iter Maximum iterations (default: 10)
 * @param tolerance Convergence tolerance in pixels² (default: 1e-12)
 * @param verbose Print convergence info (default: false)
 */
inline void kanatani_simplified(
    const Eigen::Matrix<double, 3, -1>& x,
    const Eigen::Matrix<double, 3, -1>& xp,
    const Eigen::Matrix<double, 3, 3>& F,
    Eigen::Matrix<double, 4, -1>& A,
    int max_iter = 10,
    double tolerance = 1e-12,
    bool verbose = false)
{
    const int N = x.cols();
    A.resize(4, N);
    
    // Precompute constant matrices
    Eigen::Matrix<double, 2, 3> S;
    S << 1, 0, 0,
         0, 1, 0;
    
    const Eigen::Matrix<double, 2, 2> Eh = S * F * S.transpose();
    
    // Process each point
    for (int i = 0; i < N; ++i) {
        // Initialize with measured points
        Eigen::Vector3d x0 = x.col(i);
        Eigen::Vector3d xp0 = xp.col(i);
        
        // Current estimates
        Eigen::Vector3d xk = x0;
        Eigen::Vector3d xpk = xp0;
        
        // Current displacements (initially zero)
        Eigen::Vector2d dxk = Eigen::Vector2d::Zero();
        Eigen::Vector2d dxpk = Eigen::Vector2d::Zero();
        
        double prev_error = std::numeric_limits<double>::max();
        int iter = 0;
        
        for (iter = 0; iter < max_iter; ++iter) {
            // Compute normals to current epipolar lines
            Eigen::Vector2d nk = S * F * xpk;
            Eigen::Vector2d npk = S * F.transpose() * xk;
            
            // Compute linear step size
            // λ = (x₀ᵀFx'₀ - Δxᵀ Ẽ Δx') / (nᵀn + n'ᵀn')
            double numerator = x0.dot(F * xp0) - dxk.dot(Eh * dxpk);
            double denominator = nk.dot(nk) + npk.dot(npk);
            
            // Check for degenerate case
            if (std::abs(denominator) < 1e-15) {
                // Epipolar lines are degenerate - use current estimate
                break;
            }
            
            double lambda_k = numerator / denominator;
            
            // Update displacements
            dxk = lambda_k * nk;
            dxpk = lambda_k * npk;
            
            // Update point estimates
            xk = x0 - S.transpose() * dxk;
            xpk = xp0 - S.transpose() * dxpk;
            
            // Compute reprojection error
            double error = dxk.squaredNorm() + dxpk.squaredNorm();
            
            if (verbose && i == 0) {
                std::cout << "Kanatani iter " << iter << ": error = " << error 
                         << ", lambda = " << lambda_k << std::endl;
            }
            
            // Check convergence
            if (error < tolerance) {
                break;
            }
            
            // Check if error is increasing (divergence or oscillation)
            double delta_error = prev_error - error;
            if (delta_error <= 0.0 && iter > 0) {
                // Error increased or stalled - stop
                if (delta_error < -tolerance) {
                    // Error increased significantly - revert to previous
                    xk = x0 - S.transpose() * (dxk - lambda_k * nk);
                    xpk = xp0 - S.transpose() * (dxpk - lambda_k * npk);
                }
                break;
            }
            
            // Check for very small improvement (stalled)
            if (iter > 2 && std::abs(delta_error) < tolerance * 1e-2) {
                break;
            }
            
            prev_error = error;
        }
        
        // Store corrected points
        A.col(i).head(2) = xpk.head(2);
        A.col(i).tail(2) = xk.head(2);
    }
}

//=============================================================================
// Kanatani Full Optimal Method
//=============================================================================

/**
 * @brief Kanatani full optimal correction (original formulation)
 * 
 * This is the original method from the 2008 paper. It's more complex and
 * slower than the simplified version, but included here for completeness
 * and academic interest.
 * 
 * The full method constructs a 9×9 covariance matrix and computes the
 * optimal correction using Lagrange multipliers.
 * 
 * In practice, the simplified method works just as well and is much faster.
 * 
 * @param x Points in FIRST camera, shape (3, N), homogeneous
 * @param xp Points in SECOND camera, shape (3, N), homogeneous
 * @param F Fundamental matrix such that xp'Fx = 0
 * @param A Output: corrected points [xp_corrected; x_corrected], shape (4, N)
 * @param max_iter Maximum iterations (default: 50)
 * @param tolerance Convergence tolerance in pixels² (default: 1e-12)
 * @param verbose Print convergence info (default: false)
 */
inline void kanatani_optimal(
    const Eigen::Matrix<double, 3, -1>& x,
    const Eigen::Matrix<double, 3, -1>& xp,
    const Eigen::Matrix<double, 3, 3>& F,
    Eigen::Matrix<double, 4, -1>& A,
    int max_iter = 50,
    double tolerance = 1e-12,
    bool verbose = false)
{
    const int N = x.cols();
    A.resize(4, N);
    
    // Vectorize F for easier computation
    Eigen::Matrix<double, 9, 1> u;
    u << F(0, 0), F(0, 1), F(0, 2),
         F(1, 0), F(1, 1), F(1, 2),
         F(2, 0), F(2, 1), F(2, 2);
    
    // Extract row and column blocks
    Eigen::Matrix<double, 2, 3> Fr;
    Fr << u(0), u(1), u(2),
          u(3), u(4), u(5);
    
    Eigen::Matrix<double, 2, 3> Fc;
    Fc << u(0), u(3), u(6),
          u(1), u(4), u(7);
    
    // Process each point
    for (int i = 0; i < N; ++i) {
        // Initialize - use inhomogeneous coordinates
        Eigen::Vector2d xh = x.col(i).head(2);
        Eigen::Vector2d xph = xp.col(i).head(2);
        
        // Corrections (initially zero)
        Eigen::Vector2d xt = Eigen::Vector2d::Zero();
        Eigen::Vector2d xpt = Eigen::Vector2d::Zero();
        
        double prev_error = std::numeric_limits<double>::max();
        int iter = 0;
        
        for (iter = 0; iter < max_iter; ++iter) {
            // Build the ξ vector (9×1)
            Eigen::Matrix<double, 9, 1> xi;
            xi(0) = xph(0) * xh(0) + xh(0) * xpt(0) + xph(0) * xt(0);
            xi(1) = xph(0) * xh(1) + xh(1) * xpt(0) + xph(0) * xt(1);
            xi(2) = xph(0) + xpt(0);
            xi(3) = xph(1) * xh(0) + xh(0) * xpt(1) + xph(1) * xt(0);
            xi(4) = xph(1) * xh(1) + xh(1) * xpt(1) + xph(1) * xt(1);
            xi(5) = xph(1) + xpt(1);
            xi(6) = xh(0) + xt(0);
            xi(7) = xh(1) + xt(1);
            xi(8) = 1.0;
            
            // Build the symmetric 9×9 covariance matrix V0
            Eigen::Matrix<double, 9, 9> V0 = Eigen::Matrix<double, 9, 9>::Zero();
            
            // Diagonal elements
            V0(0, 0) = xph(0) * xph(0) + xh(0) * xh(0);
            V0(1, 1) = xph(0) * xph(0) + xh(1) * xh(1);
            V0(2, 2) = 1.0;
            V0(3, 3) = xph(1) * xph(1) + xh(0) * xh(0);
            V0(4, 4) = xph(1) * xph(1) + xh(1) * xh(1);
            V0(5, 5) = 1.0;
            V0(6, 6) = 1.0;
            V0(7, 7) = 1.0;
            V0(8, 8) = 0.0;  // Not used, but set for completeness
            
            // Off-diagonal elements (symmetric matrix)
            V0(0, 1) = V0(1, 0) = xh(0) * xh(1);
            V0(0, 2) = V0(2, 0) = xh(0);
            V0(0, 3) = V0(3, 0) = xph(0) * xph(1);
            V0(0, 6) = V0(6, 0) = xph(0);
            
            V0(1, 2) = V0(2, 1) = xh(1);
            V0(1, 4) = V0(4, 1) = xph(0) * xph(1);
            V0(1, 7) = V0(7, 1) = xph(0);
            
            V0(3, 4) = V0(4, 3) = xh(0) * xh(1);
            V0(3, 5) = V0(5, 3) = xh(0);
            V0(3, 6) = V0(6, 3) = xph(1);
            
            V0(4, 5) = V0(5, 4) = xh(1);
            V0(4, 7) = V0(7, 4) = xph(1);
            
            // Compute step size
            double denominator = u.dot(V0 * u);
            
            if (std::abs(denominator) < 1e-15) {
                // Degenerate case
                break;
            }
            
            double s = u.dot(xi) / denominator;
            
            // Update corrections
            xpt = s * (Fr.block<2, 2>(0, 0) * xh + Fr.col(2));
            xt = s * (Fc.block<2, 2>(0, 0) * xph + Fc.col(2));
            
            // Compute reprojection error
            double error = xt.squaredNorm() + xpt.squaredNorm();
            
            if (verbose && i == 0) {
                std::cout << "Kanatani full iter " << iter << ": error = " << error 
                         << ", s = " << s << std::endl;
            }
            
            // Check convergence
            if (error < tolerance) {
                break;
            }
            
            // Check if error is increasing
            double delta_error = prev_error - error;
            if (delta_error <= 0.0 && iter > 0) {
                // Error increased or stalled
                break;
            }
            
            // Check for very small improvement
            if (iter > 2 && std::abs(delta_error) < tolerance * 1e-2) {
                break;
            }
            
            // Update estimates
            xh = xp.col(i).head(2) - xpt;
            xph = x.col(i).head(2) - xt;
            prev_error = error;
        }
        
        // Final corrected points
        Eigen::Vector2d x_final = x.col(i).head(2) - xt;
        Eigen::Vector2d xp_final = xp.col(i).head(2) - xpt;
        
        // Store result
        A.col(i).head(2) = xp_final;
        A.col(i).tail(2) = x_final;
    }
}

//=============================================================================
// Kanatani with Adaptive Damping (Experimental)
//=============================================================================

/**
 * @brief Kanatani simplified with adaptive damping for better convergence
 * 
 * This variant adds Levenberg-Marquardt style damping to improve convergence
 * in difficult configurations. Experimental but can help with unstable cases.
 * 
 * @param x Points in FIRST camera, shape (3, N), homogeneous
 * @param xp Points in SECOND camera, shape (3, N), homogeneous
 * @param F Fundamental matrix such that xp'Fx = 0
 * @param A Output: corrected points [xp_corrected; x_corrected], shape (4, N)
 * @param max_iter Maximum iterations (default: 20)
 * @param tolerance Convergence tolerance in pixels² (default: 1e-12)
 * @param initial_damping Initial damping factor (default: 1e-6)
 * @param verbose Print convergence info (default: false)
 */
inline void kanatani_damped(
    const Eigen::Matrix<double, 3, -1>& x,
    const Eigen::Matrix<double, 3, -1>& xp,
    const Eigen::Matrix<double, 3, 3>& F,
    Eigen::Matrix<double, 4, -1>& A,
    int max_iter = 20,
    double tolerance = 1e-12,
    double initial_damping = 1e-6,
    bool verbose = false)
{
    const int N = x.cols();
    A.resize(4, N);
    
    // Precompute constant matrices
    Eigen::Matrix<double, 2, 3> S;
    S << 1, 0, 0,
         0, 1, 0;
    
    const Eigen::Matrix<double, 2, 2> Eh = S * F * S.transpose();
    
    // Process each point
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d x0 = x.col(i);
        Eigen::Vector3d xp0 = xp.col(i);
        
        Eigen::Vector3d xk = x0;
        Eigen::Vector3d xpk = xp0;
        
        Eigen::Vector2d dxk = Eigen::Vector2d::Zero();
        Eigen::Vector2d dxpk = Eigen::Vector2d::Zero();
        
        double damping = initial_damping;
        double prev_error = std::numeric_limits<double>::max();
        
        for (int iter = 0; iter < max_iter; ++iter) {
            Eigen::Vector2d nk = S * F * xpk;
            Eigen::Vector2d npk = S * F.transpose() * xk;
            
            double numerator = x0.dot(F * xp0) - dxk.dot(Eh * dxpk);
            double denominator = nk.dot(nk) + npk.dot(npk);
            
            // Add damping term
            denominator += damping * (nk.dot(nk) + npk.dot(npk));
            
            if (std::abs(denominator) < 1e-15) {
                break;
            }
            
            double lambda_k = numerator / denominator;
            
            // Try this step
            Eigen::Vector2d dxk_new = lambda_k * nk;
            Eigen::Vector2d dxpk_new = lambda_k * npk;
            
            double error = dxk_new.squaredNorm() + dxpk_new.squaredNorm();
            
            if (verbose && i == 0) {
                std::cout << "Kanatani damped iter " << iter 
                         << ": error = " << error 
                         << ", damping = " << damping << std::endl;
            }
            
            // Accept or reject step based on error
            if (error < prev_error || iter == 0) {
                // Good step - accept and decrease damping
                dxk = dxk_new;
                dxpk = dxpk_new;
                xk = x0 - S.transpose() * dxk;
                xpk = xp0 - S.transpose() * dxpk;
                
                damping = std::max(damping * 0.5, 1e-12);
                
                if (error < tolerance) {
                    break;
                }
                
                if (iter > 2 && std::abs(prev_error - error) < tolerance * 1e-2) {
                    break;
                }
                
                prev_error = error;
            } else {
                // Bad step - reject and increase damping
                damping = std::min(damping * 2.0, 1e3);
                
                if (damping > 1e2) {
                    // Damping too high - probably stuck
                    break;
                }
            }
        }
        
        A.col(i).head(2) = xpk.head(2);
        A.col(i).tail(2) = xk.head(2);
    }
}

//=============================================================================
// Convenience Wrappers
//=============================================================================

/**
 * @brief Single-point Kanatani simplified
 */
inline Eigen::Vector4d kanatani_simplified(
    const Eigen::Vector3d& x,
    const Eigen::Vector3d& xp,
    const Eigen::Matrix3d& F,
    int max_iter = 10,
    double tolerance = 1e-12)
{
    Eigen::Matrix<double, 3, 1> x_mat = x;
    Eigen::Matrix<double, 3, 1> xp_mat = xp;
    Eigen::Matrix<double, 4, Eigen::Dynamic> result(4,1);
    
    kanatani_simplified(x_mat, xp_mat, F, result, max_iter, tolerance);
    return result.col(0);
}

/**
 * @brief Single-point Kanatani optimal
 */
inline Eigen::Vector4d kanatani_optimal(
    const Eigen::Vector3d& x,
    const Eigen::Vector3d& xp,
    const Eigen::Matrix3d& F,
    int max_iter = 50,
    double tolerance = 1e-12)
{
    Eigen::Matrix<double, 3, 1> x_mat = x;
    Eigen::Matrix<double, 3, 1> xp_mat = xp;
    Eigen::Matrix<double, 4, Eigen::Dynamic> result(4,1);
    
    kanatani_optimal(x_mat, xp_mat, F, result, max_iter, tolerance);
    return result.col(0);
}

/**
 * @brief Single-point Kanatani damped
 */
inline Eigen::Vector4d kanatani_damped(
    const Eigen::Vector3d& x,
    const Eigen::Vector3d& xp,
    const Eigen::Matrix3d& F,
    int max_iter = 20,
    double tolerance = 1e-12,
    double initial_damping = 1e-6)
{
    Eigen::Matrix<double, 3, 1> x_mat = x;
    Eigen::Matrix<double, 3, 1> xp_mat = xp;
    Eigen::Matrix<double, 4, Eigen::Dynamic> result(4,1);
    
    kanatani_damped(x_mat, xp_mat, F, result, max_iter, tolerance, initial_damping);
    return result.col(0);
}

//=============================================================================
// Default Method (Recommended)
//=============================================================================

/**
 * @brief Kanatani triangulation (uses simplified method by default)
 * 
 * This is the recommended entry point. Uses the simplified method which
 * is fast and works well in practice.
 */
inline void kanatani_triangulate(
    const Eigen::Matrix<double, 3, -1>& x,
    const Eigen::Matrix<double, 3, -1>& xp,
    const Eigen::Matrix<double, 3, 3>& F,
    Eigen::Matrix<double, 4, -1>& A,
    int max_iter = 10,
    double tolerance = 1e-12)
{
    kanatani_simplified(x, xp, F, A, max_iter, tolerance);
}

/**
 * @brief Single-point Kanatani (default)
 */
inline Eigen::Vector4d kanatani_triangulate(
    const Eigen::Vector3d& x,
    const Eigen::Vector3d& xp,
    const Eigen::Matrix3d& F,
    int max_iter = 10,
    double tolerance = 1e-12)
{
    return kanatani_simplified(x, xp, F, max_iter, tolerance);
}

} // namespace triangulation
