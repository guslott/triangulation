/*
    Fast Optimal Triangulation
    Dr. Gus K Lott

    An implementation of 2x2 SVD meant for performance and portability.
    Uses Jacobi rotations for numerical stability.
    A = U*D*V'


    MIT License

    Copyright (c) 2021 Dr. Gus K. Lott, guslott@gmail.com

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/
#pragma once
#include <Eigen/Dense>
#include <cmath>

class SVD2x2_Jacobi 
{

  public: 
    SVD2x2_Jacobi(
        const Eigen::Matrix<double,2,2> & A);

    //Results are stored in this data structure
    const Eigen::Matrix<double,2,2> & U() const { return U_; }
    const Eigen::Matrix<double,2,2> & V() const { return V_; }
    const Eigen::Matrix<double,1,2> & d() const { return d_; }
    const double d(int idx) const { return d_(idx); }

  private:
    Eigen::Matrix<double,2,2> U_, V_; 
    Eigen::Matrix<double,1,2> d_;

    //The technique for solving U/V from the symmetric system A'*A and A*A'
    static Eigen::Matrix<double,2,2>
    jacobi_rotation(const double alpha, const double beta, const double gamma);

    //When U/V are solved, it is up to an unknown permutation of diag([a,b])
    //Either off diagonal, out of amplitude order, and/or negative values.
    //Solve all this to canonical form via permutations and sign flips of the columns of U/V
    double clean_permutations_and_signs(const Eigen::Matrix<double,2,2> & Dt);

    //In the case of patholotical matrices where the two singular valus are very close
    // the jacobi rotations (as implemented) are not stable.  Backup to ATAN method (slower)
    void atanUV(const Eigen::Matrix<double,2,2> & A);
};


SVD2x2_Jacobi::SVD2x2_Jacobi(
    const Eigen::Matrix<double,2,2> & A)
{
    //Check for simple cases
    //Null Matrix
    if((A(0,0)==0)&(A(1,1)==0)&(A(1,0)==0)&(A(0,1)==0))
    {
        U_ = Eigen::Matrix<double,2,2>::Identity();
        V_ = Eigen::Matrix<double,2,2>::Identity();
        d_(0) = d_(1) = 0;
        return; 
    }
    //Matrix is already diagonal
    if((A(0,1)==0)&(A(1,0)==0))
    {
        U_ = Eigen::Matrix<double,2,2>::Identity();
        V_ = Eigen::Matrix<double,2,2>::Identity();
        clean_permutations_and_signs(A);
        return;
    }
    //Anti-diagonal matrix
    if((A(0,0)==0)&(A(1,1)==0))
    {
        U_ = Eigen::Matrix<double,2,2>::Identity();
        V_ = Eigen::Matrix<double,2,2>::Identity();
        clean_permutations_and_signs(A);
        return;
    }

    //Solves the SVD 2x2 directly using Jacobi Rotations
    //Factors the Matrix in the form: A = U*D*V'
    //First solve for U & V, then solve for D using U'*A*V

    //General case solution:
    // A = U*D*V' - U,V orthonormal matrices
    // Solve for V from A'*A = V*D*D*V' - a symmetric eigensystem
    // C = A'*A; = [alpha, gamma; 
    //              gamma, beta]
    const double gamma_v = A(0,0)*A(0,1) + A(1,0)*A(1,1);
    const double alpha_v = A(0,0)*A(0,0) + A(1,0)*A(1,0);
    const double beta_v = A(0,1)*A(0,1) + A(1,1)*A(1,1);
    V_ = jacobi_rotation(alpha_v,beta_v,gamma_v);

    //Solve for U from C = A*A' = U*D*D*U' 
    const double gamma_u = A(0,0)*A(1,0) + A(1,1)*A(0,1);
    const double alpha_u = A(0,0)*A(0,0) + A(0,1)*A(0,1);
    const double beta_u = A(1,0)*A(1,0) + A(1,1)*A(1,1);
    U_ = jacobi_rotation(alpha_u,beta_u,gamma_u);

    if((gamma_u == 0)&(gamma_v==0))
    {
        //Special case where singular values are exactly equal
        //The Matrix is exactly a scaled orthogonal matrix
        //Slightly off of this is a pathalogical case - atan kicks in
        const double det_A = A(0,0)*A(1,1) - A(0,1)*A(1,0);
        d_(0) = d_(1) = sqrt(abs(det_A));
        U_ = A/d_(0);
        //V already set to identity above
        return;
    }

    //Compute the singular values and ordering
    const double residual = clean_permutations_and_signs(U_.transpose()*A*V_);
    
    //If residual is above a threshold, fall back on atan method.
    // can happen if the singular values are very close but not exactly the same
    if(residual > 1e-12)
    {
        atanUV(A);
        clean_permutations_and_signs(U_.transpose()*A*V_);
    }

    return;
}

double
SVD2x2_Jacobi::clean_permutations_and_signs(
    const Eigen::Matrix<double,2,2> & Dt)
{
    //Given Dt = U' * A * V, the matrix may be anti-diagonal or diagonal.
    // the singular values may be negative, and out of order in amplitude
    // Clean all this to canonical ordering and sign

    //Find the maximum amplitude element
    double maxel = abs(Dt(0,0));
    int maxi = 0;
    if(abs(Dt(0,1)) > maxel)
    {
        maxel = abs(Dt(0,1));
        maxi = 1;
    }
    if(abs(Dt(1,0)) > maxel)
    {
        maxel = abs(Dt(1,0));
        maxi = 2;
    }
    if(abs(Dt(1,1)) > maxel)
    {
        maxel = abs(Dt(1,1));
        maxi = 3;
    }

    //Fix permutation and ordering of amplitude.
    double residual = 0;
    if(maxi == 0) //Correct ordering, max is upper left
    {
        d_(0) = Dt(0,0);
        d_(1) = Dt(1,1);
        residual = abs(Dt(0,1)) + abs(Dt(1,0));
    }
    if(maxi == 1)
    {
        //Maxel is top right, swap columns of V
        std::swap(V_(0,0),V_(0,1));
        std::swap(V_(1,0),V_(1,1));
        d_(0) = Dt(0,1);
        d_(1) = Dt(1,0);
        residual = abs(Dt(0,0)) + abs(Dt(1,1));
    }
    if(maxi == 2)
    {
        //maxel is bottom left, swap columns of U
        std::swap(U_(0,0),U_(0,1));
        std::swap(U_(1,0),U_(1,1));
        d_(0) = Dt(1,0);
        d_(1) = Dt(0,1);
        residual = abs(Dt(0,0)) + abs(Dt(1,1));
    }
    if(maxi == 3)
    {
        //Maxel is in bottom right, swap columns of U and V
        std::swap(U_(0,0),U_(0,1));
        std::swap(U_(1,0),U_(1,1));
        std::swap(V_(0,0),V_(0,1));
        std::swap(V_(1,0),V_(1,1));
        d_(0) = Dt(1,1);
        d_(1) = Dt(0,0);
        residual = abs(Dt(0,1)) + abs(Dt(1,0));
    }

    //Fix sign of singular values using columns of V
    if(d_(0) < 0)
    {
        V_(0,0) = -V_(0,0);
        V_(1,0) = -V_(1,0);
        d_(0) = -d_(0);
    }
    if(d_(1) < 0)
    {
        V_(0,1) = -V_(0,1);
        V_(1,1) = -V_(1,1);
        d_(1) = -d_(1);
    }

    return residual;
}

Eigen::Matrix<double,2,2>
SVD2x2_Jacobi::jacobi_rotation(
    const double alpha,
    const double beta,
    const double gamma)
{
    //For the symmetric matrix: C = [alpha, gamma; gamma, beta]
    //Compute the rotation R that diagonalizes it using the Jacobi method
    double t,c;
    if(gamma == 0)
    {
        t = 0;
        c = 1;
    }else
    {
        //These "sqrt" are the major compute cost of the algorithm
        const double zeta = (beta - alpha)/(2.0*gamma); //zeta = cot(2*theta)
        t = 1.0/(abs(zeta)+sqrt(1.0 + zeta*zeta)); //double angle identity
        if(zeta<0) t = -t;  //t = tan(theta)
        c = 1.0/sqrt(1.0 + t*t);//c = cos(theta)
    }
    const double s = c*t; //sin(theta)
    Eigen::Matrix<double,2,2> R;
    R(0,0) = c; R(0,1) = s;
    R(1,0) =-s; R(1,1) = c;
    return R;
}

void
SVD2x2_Jacobi::atanUV(
    const Eigen::Matrix<double,2,2> & A)
{
    //A backup in pathalogical cases where the Jacobi method has issues
    //The trig function calls here are computationally costly
    const double thp = atan2(-A(1,0)-A(0,1),A(0,0)-A(1,1)); //angle u+v
    const double thm = atan2( A(0,1)-A(1,0),A(0,0)+A(1,1)); //angle u-v
    const double u = (thp+thm)/2;
    const double v = (thp-thm)/2;

    U_(0,0) =   cos(u); U_(0,1) = sin(u);
    U_(1,0) = -U_(0,1); U_(1,1) = U_(0,0);

    V_(0,0) =   cos(v); V_(0,1) = sin(v);
    V_(1,0) = -V_(0,1); V_(1,1) = V_(0,0);
}
