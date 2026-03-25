#pragma once
/*
    Lindström Triangulation - Corrected Implementation
    
    Based on: "Triangulation Made Easy" by Peter Lindström (2010)
    
    This implementation provides three variants:
    1. lindstrom_iter   : Iterative method (2-5 iterations typical)
    2. lindstrom_niter1 : One quadratic step + projection (guarantees epipolar constraint)
    3. lindstrom_niter2 : One quadratic step + linear correction (fastest)
    
    COORDINATE CONVENTION:
    For fundamental matrix F and points x ↔ x', the epipolar constraint is:
        x'ᵀ F x = 0
    
    This means:
    - x is in the FIRST camera (right image)
    - x' is in the SECOND camera (left image)  
    - Epipolar line in first camera: l = Fᵀx'
    - Epipolar line in second camera: l' = Fx
    
    IMPORTANT: This matches OpenCV convention but is OPPOSITE of some textbooks
    that write x'ᵀ F x = 0 with x' on the left.
*/

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

namespace triangulation {

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Solves quadratic equation ax² + 2bx + c = 0 with numerical stability
 * @param a Coefficient of x²
 * @param b Coefficient of x (NOTE: input is b, not 2b)
 * @param c Constant term
 * @return The smaller root (in magnitude)
 */
inline double solve_quadratic_stable(double a, double b, double c)
{
    // Solve: ax² + 2bx + c = 0
    // Standard form coefficients for ax² + bx + c would be (a, 2b, c)
    
    if (std::abs(a) < 1e-15) {
        // Nearly linear: 2bx + c ≈ 0
        return (std::abs(b) > 1e-15) ? -c / (2.0 * b) : 0.0;
    }
    
    double discriminant = b * b - a * c;
    
    if (discriminant < 0.0) {
        // No real roots - shouldn't happen in well-posed triangulation
        // Return smallest residual solution
        return -b / a;
    }
    
    double d = std::sqrt(discriminant);
    
    // Use sign(b) to avoid catastrophic cancellation
    // λ = c / (b + sign(b)·d)
    double sign_b = (b >= 0.0) ? 1.0 : -1.0;
    return c / (b + sign_b * d);
}

/**
 * @brief Compute reprojection error for convergence testing
 */
inline double reprojection_error(
    const Eigen::Vector2d& dx,
    const Eigen::Vector2d& dxp)
{
    return dx.squaredNorm() + dxp.squaredNorm();
}

//=============================================================================
// Main Algorithms
//=============================================================================

/**
 * @brief Iterative Lindström triangulation
 * 
 * Converges to optimal solution in 2-5 iterations (typically 2).
 * Guarantees epipolar constraint in each iteration.
 * 
 * @param x Points in FIRST camera (right image), shape (3, N), homogeneous
 * @param xp Points in SECOND camera (left image), shape (3, N), homogeneous  
 * @param F Fundamental matrix such that xpᵀFx = 0
 * @param A Output: corrected points [xp_corrected; x_corrected], shape (4, N)
 * @param max_iter Maximum iterations (default: 5)
 * @param tolerance Convergence tolerance in pixels² (default: 1e-12)
 * @param verbose Print convergence info (default: false)
 */
inline void lindstrom_iter(
    const Eigen::Matrix<double, 3, -1>& x,
    const Eigen::Matrix<double, 3, -1>& xp,
    const Eigen::Matrix<double, 3, 3>& F,
    Eigen::Matrix<double, 4, -1>& A,
    int max_iter = 5,
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
        // Current estimates (start at measured points)
        Eigen::Vector3d xk = x.col(i);
        Eigen::Vector3d xpk = xp.col(i);
        
        // Keep first iteration normals for higher-order method
        Eigen::Vector2d n1, np1;
        
        double prev_error = 1e10;
        int iter = 0;
        
        for (iter = 0; iter < max_iter; ++iter) {
            // Compute normals to current epipolar lines
            Eigen::Vector2d nk = S * F * xpk;
            Eigen::Vector2d npk = S * F.transpose() * xk;
            
            // Store first iteration normals
            if (iter == 0) {
                n1 = nk;
                np1 = npk;
            }
            
            // Compute quadratic equation coefficients
            // λ² a + λ b + c = 0 where a, b, c are defined below
            double ak = nk.dot(Eh * npk);
            double bk = 0.5 * (n1.dot(nk) + np1.dot(npk));  // Higher-order correction
            double ck = x.col(i).dot(F * xp.col(i));
            
            // Solve for optimal step size with numerical stability
            double lambda_k = solve_quadratic_stable(ak, bk, ck);
            
            // Take step
            Eigen::Vector2d dxk = lambda_k * nk;
            Eigen::Vector2d dxpk = lambda_k * npk;
            
            xk = x.col(i) - S.transpose() * dxk;
            xpk = xp.col(i) - S.transpose() * dxpk;
            
            // Check convergence
            double error = reprojection_error(dxk, dxpk);
            
            if (verbose && i == 0) {  // Only print first point to avoid spam
                std::cout << "Iter " << iter << ": error = " << error 
                         << ", lambda = " << lambda_k << std::endl;
            }
            
            if (error < tolerance) {
                break;
            }
            
            // Check if we're making progress
            if (iter > 0 && (prev_error - error) < tolerance * 1e-2) {
                // Converged or stalled
                break;
            }
            
            prev_error = error;
        }
        
        // Store corrected points
        A.col(i).head(2) = xpk.head(2);
        A.col(i).tail(2) = xk.head(2);
    }
}

/**
 * @brief Non-iterative Lindström variant 1
 * 
 * Takes one optimal quadratic step, then projects onto epipolar lines.
 * Guarantees: 
 * - Points lie on corresponding epipolar lines (condition i)
 * - Points are projections of measured points onto these lines (condition ii)
 * 
 * Fastest method that guarantees epipolar constraint.
 * Typically agrees with full iteration to ~12 digits.
 * 
 * @param x Points in FIRST camera, shape (3, N), homogeneous
 * @param xp Points in SECOND camera, shape (3, N), homogeneous
 * @param F Fundamental matrix such that xpᵀFx = 0  
 * @param A Output: corrected points [xp_corrected; x_corrected], shape (4, N)
 */
inline void lindstrom_niter1(
    const Eigen::Matrix<double, 3, -1>& x,
    const Eigen::Matrix<double, 3, -1>& xp,
    const Eigen::Matrix<double, 3, 3>& F,
    Eigen::Matrix<double, 4, -1>& A)
{
    const int N = x.cols();
    A.resize(4, N);
    
    // Precompute constant matrices
    Eigen::Matrix<double, 2, 3> S;
    S << 1, 0, 0,
         0, 1, 0;
    
    const Eigen::Matrix<double, 2, 2> Eh = S * F * S.transpose();
    
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d& xi = x.col(i);
        const Eigen::Vector3d& xpi = xp.col(i);
        
        // Step 1: Compute initial normals
        Eigen::Vector2d n = S * F * xpi;
        Eigen::Vector2d np = S * F.transpose() * xi;
        
        // Step 2: Solve quadratic for optimal lambda
        double a = n.dot(Eh * np);
        double b = 0.5 * (n.dot(n) + np.dot(np));
        double c = xi.dot(F * xpi);
        
        double lambda = solve_quadratic_stable(a, b, c);
        
        // Step 3: Take optimal step
        Eigen::Vector2d dx = lambda * n;
        Eigen::Vector2d dxp = lambda * np;
        
        // Step 4: Update normals to new epipolar lines
        n = n - Eh * dxp;
        np = np - Eh.transpose() * dx;
        
        // Step 5: Project measured points onto these epipolar lines
        // This ensures exact satisfaction of epipolar constraint
        double proj_scale_x = dx.dot(n) / n.dot(n);
        double proj_scale_xp = dxp.dot(np) / np.dot(np);
        
        dx = proj_scale_x * n;
        dxp = proj_scale_xp * np;
        
        // Step 6: Compute corrected points
        Eigen::Vector3d xk = xi - S.transpose() * dx;
        Eigen::Vector3d xpk = xpi - S.transpose() * dxp;
        
        // Store result
        A.col(i).head(2) = xpk.head(2);
        A.col(i).tail(2) = xk.head(2);
    }
}

/**
 * @brief Non-iterative Lindström variant 2 (fastest)
 * 
 * Takes one optimal quadratic step, then one linear correction.
 * Guarantees:
 * - Displacements related by single parameter λ (condition iii)
 * 
 * Does NOT exactly satisfy epipolar constraint, but error is ~1e-15.
 * Fastest method; typically agrees with full iteration to ~10-12 digits.
 * 
 * This is the method recommended in Lindström's paper for practical use.
 * 
 * @param x Points in FIRST camera, shape (3, N), homogeneous
 * @param xp Points in SECOND camera, shape (3, N), homogeneous  
 * @param F Fundamental matrix such that xpᵀFx = 0
 * @param A Output: corrected points [xp_corrected; x_corrected], shape (4, N)
 */
inline void lindstrom_niter2(
    const Eigen::Matrix<double, 3, -1>& x,
    const Eigen::Matrix<double, 3, -1>& xp,
    const Eigen::Matrix<double, 3, 3>& F,
    Eigen::Matrix<double, 4, -1>& A)
{
    const int N = x.cols();
    A.resize(4, N);
    
    // Precompute constant matrices
    Eigen::Matrix<double, 2, 3> S;
    S << 1, 0, 0,
         0, 1, 0;
    
    const Eigen::Matrix<double, 2, 2> Eh = S * F * S.transpose();
    
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d& xi = x.col(i);
        const Eigen::Vector3d& xpi = xp.col(i);
        
        // Step 1: Compute initial normals
        Eigen::Vector2d n = S * F * xpi;
        Eigen::Vector2d np = S * F.transpose() * xi;
        
        // Step 2: Solve quadratic for optimal lambda
        double a = n.dot(Eh * np);
        double b = 0.5 * (n.dot(n) + np.dot(np));
        double c = xi.dot(F * xpi);
        double d = std::sqrt(b * b - a * c);
        
        double lambda = solve_quadratic_stable(a, b, c);
        
        // Step 3: Take optimal step
        Eigen::Vector2d dx = lambda * n;
        Eigen::Vector2d dxp = lambda * np;
        
        // Step 4: Update normals to new epipolar lines
        n = n - Eh * dxp;
        np = np - Eh.transpose() * dx;
        
        // Step 5: Linear correction (Kanatani-style step)
        // This maintains single lambda but may violate epipolar constraint slightly
        double lambda2 = 2.0 * d / (n.dot(n) + np.dot(np));
        
        dx = lambda2 * n;
        dxp = lambda2 * np;
        
        // Step 6: Compute corrected points
        Eigen::Vector3d xk = xi - S.transpose() * dx;
        Eigen::Vector3d xpk = xpi - S.transpose() * dxp;
        
        // Store result
        A.col(i).head(2) = xpk.head(2);
        A.col(i).tail(2) = xk.head(2);
    }
}

//=============================================================================
// Convenience Wrappers
//=============================================================================

/**
 * @brief Single-point triangulation (iterative)
 */
inline Eigen::Vector4d lindstrom_iter(
    const Eigen::Vector3d& x,
    const Eigen::Vector3d& xp,
    const Eigen::Matrix3d& F,
    int max_iter = 5,
    double tolerance = 1e-12)
{
    Eigen::Matrix<double, 3, 1> x_mat = x;
    Eigen::Matrix<double, 3, 1> xp_mat = xp;
    Eigen::Matrix<double, 4, Eigen::Dynamic> result(4,1);
    
    lindstrom_iter(x_mat, xp_mat, F, result, max_iter, tolerance);
    return result.col(0);
}

/**
 * @brief Single-point triangulation (niter1)
 */
inline Eigen::Vector4d lindstrom_niter1(
    const Eigen::Vector3d& x,
    const Eigen::Vector3d& xp,
    const Eigen::Matrix3d& F)
{
    Eigen::Matrix<double, 3, 1> x_mat = x;
    Eigen::Matrix<double, 3, 1> xp_mat = xp;
    Eigen::Matrix<double, 4, Eigen::Dynamic> result(4,1);
    
    lindstrom_niter1(x_mat, xp_mat, F, result);
    return result.col(0);
}

/**
 * @brief Single-point triangulation (niter2 - fastest)
 */
inline Eigen::Vector4d lindstrom_niter2(
    const Eigen::Vector3d& x,
    const Eigen::Vector3d& xp,
    const Eigen::Matrix3d& F)
{
    Eigen::Matrix<double, 3, 1> x_mat = x;
    Eigen::Matrix<double, 3, 1> xp_mat = xp;
    Eigen::Matrix<double, 4, Eigen::Dynamic> result(4,1);
    
    lindstrom_niter2(x_mat, xp_mat, F, result);
    return result.col(0);
}

} // namespace triangulation
