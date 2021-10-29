/*
	Fast Optimal Triangulation

	Given a pair of 2D image correspondences and a fundamental matrix
	describing the projective relationship of the views, the triangulation
	algorithm finds the nearest image points which perfectly satisfies
	the projective relationship so that back projected rays perfectly intersect
	in space.

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

#include "svd2x2_lott.h"

template <int PORDER>
double 
polyval(
	const double * p,
	const double x)
{
	//Evaluate polynomial using Horner's recursion
	double px = p[0];
	for(int i = 1; i <= PORDER; ++i)
	{
		px = px*x + p[i];
	}
	return px;
}

template <int PORDER>
void poly_derivative(
	const double * p, 
	double * dp)
{
	for(int i=0; i < PORDER; i++)
	{
		dp[i] = p[i]*(PORDER-i);
	}
}

template <int HORDER>
double householder_step_from_origin(
	const double * p)
{
	
	//Householder Methods for polynomial root finding "from the origin"
	//Only need the first HORDER+1 coefficients of the polynomial
	//Converges to the root at an HORDER+1 rate.  Does not iterate
	const double & k0 = p[6];
	const double & k1 = p[5];
	const double & k2 = p[4];
	const double & k3 = p[3];
	const double & k4 = p[2];
	const double & k5 = p[1];
	const double & k6 = p[0];

	if(HORDER == 1)
	{
		//Newton-Raphson's Method
		const double num = -k0;
		const double den =  k1;
		return (num/den);
	}
	if(HORDER == 2)
	{
		//Halley's method
		const double num = -k0*k1;
		const double den = (k1*k1 - k0*k2);
		return (num/den);
	}
	//Higher order methods
	if(HORDER == 3)
	{
		const double num = -(k0*k1*k1 - k0*k0*k2);
		const double den =  (k1*k1*k1 - 2*k0*k1*k2 + k0*k0*k3);
		return (num/den);
	}
	if(HORDER == 4)
	{
		const double num = -(k0*k1*k1*k1 - 2*k0*k0*k1*k2 + k0*k0*k0*k3);
		const double den =  (k1*k1*k1*k1 - 3*k0*k1*k1*k2 + k0*k0*k2*k2 + 2*k0*k0*k1*k3 - k0*k0*k0*k4);
		return (num/den);
	}
	if(HORDER == 5)
	{
		const double num = -k0*k1*k1*k1*k1 + 3*k0*k0*k1*k1*k2 -   k0*k0*k0*k2*k2 - 2*k0*k0*k0*k1*k3 + k0*k0*k0*k0*k4;
		const double den =  k1*k1*k1*k1*k1 - 4*k0*k1*k1*k1*k2 + 3*k0*k0*k1*k2*k2 - 2*k0*k0*k0*k1*k4 + k0*k0*k0*k0*k5 
			            + 3*k0*k0*k1*k1*k3 - 2*k0*k0*k0*k2*k3;
		return (num/den);
	}

	return 0; //is an error, no update to the root
}

template <int NCOEF>
void lott_poly6_cx(
	const double a, const double b, 
	const double c, const double d, const double e, const double f, const double g,
	double * p)// p assumed to have 7 doubles of space
{	
	//The roots of this polynomial must be multiplied by -c in order to correspond to the solution
	//This polynomial is p6(-c*x)/c^4 if p6(x) is the polynomial from the paper Table 1
	const double a2 = a*a;
	const double b2 = b*b;
	const double c2 = c*c;
	const double d2 = d*d;
	const double e2 = e*e;
	const double f2 = f*f;
	const double nu2 = c2 + d2 + e2 + f2;
	const double rho = a*(c2-e2) + b*(d2-f2);

	if(NCOEF>=1) p[6] = g; //x^0 term
	if(NCOEF>=2) p[5] = -(6*a*g + 2*nu2);
	if(NCOEF>=3) p[4] = (3*rho + g*(13*a2-2*b2) + 10*a*nu2);
	if(NCOEF>=4) p[3] = -(8*(2*c2-e2)*(a2-b2) + 4*a*g*(3*a2+b2) + 16*a2*nu2);
	if(NCOEF>=5) p[2] = (a*(a2-b2)*(25*c2-e2) + rho*(4*a2-b2) + g*(4*a2*a2+b2*b2)+8*a2*a*nu2);
	if(NCOEF>=6) p[1] = -(4*c2*(5*a2-b2)*(a2-b2));
	if(NCOEF>=7) p[0] = (4*a*c2*(a2-b2)*(a2-b2)); //x^6 term
}

template <int HORDER>
double householder_step_from_origin(
	const double a, const double b, 
	const double c, const double d, const double e, const double f, const double g)
{
	//Simply compute the householder step of the specified order
	// This is the approximate triangulation.  Only needs a few coefficients
	double p[7];
	lott_poly6_cx<HORDER+1>(a,b,c,d,e,f,g,p);
	return -c*householder_step_from_origin<HORDER>(p); //-c factor due to change of variables
}

double 
full_root_iterative(
	const double a, const double b, 
	const double c, const double d, const double e, const double f, const double g) 
{
	double p[7];
	lott_poly6_cx<7>(a,b,c,d,e,f,g,p);
	//First step from zero using a given householder method is inexpensive
	double x = householder_step_from_origin<4>(p);

    double dp[6];
    poly_derivative<6>(p,dp);
	//Iteratively converge
    for(int loops = 0; loops < 5; ++loops)
    {
    	const double px = polyval<6>(p,x);
    	if(abs(px) < 1e-15) break;
        const double dpx = polyval<5>(dp,x);
        x += (-px/dpx); //Newton-Raphson Iteration (first order householder method)
    }
    return -c*x;//Poly above had change of variables to condition near zero
}

//Distance Metric - return squared reprojection error
// This is a quadratic upgrade to the linear sampson distance
double
lott_distance_quadratic(
	const Eigen::Matrix<double,3,3> & F,
	const Eigen::Matrix<double,4,1> & A)
{
	//Singular value, a - This factor is point independent, 
	// may be extracted for speed - compiler will probably do this for you
	const double r1 = sqrt((F(0,0)+F(1,1))*(F(0,0)+F(1,1)) + (F(0,1) - F(1,0))*(F(0,1) - F(1,0)));
	const double r2 = sqrt((F(0,0)-F(1,1))*(F(0,0)-F(1,1)) + (F(0,1) + F(1,0))*(F(0,1) + F(1,0)));
	const double a = 0.5*(r1 + r2);

	//Point dependent parameters:
	//parameter nu^2
	const double Fr0x0 = F(0,0)*A(0) + F(0,1)*A(1) + F(0,2);
	const double Fr1x0 = F(1,0)*A(0) + F(1,1)*A(1) + F(1,2);
	const double Fc0x1 = F(0,0)*A(2) + F(1,0)*A(3) + F(2,0);
	const double Fc1x1 = F(0,1)*A(2) + F(1,1)*A(3) + F(2,1);
	const double nu2 = Fr0x0*Fr0x0 + Fr1x0*Fr1x0 + Fc0x1*Fc0x1 + Fc1x1*Fc1x1;

	//parameter g
	const double g = 2*(Fr0x0*A(2) + Fr1x0*A(3) + F(2,0)*A(0) + F(2,1)*A(1) + F(2,2));

	//Compute the squared reprojection error
	const double den = (6*a*g + 2*nu2);
	const double eps2 = g*g*nu2/(den*den);
	return eps2;
}

void lott_triangulate(
	const Eigen::Matrix<double,4,-1> & A,
	const Eigen::Matrix<double,3,3> & F,
	Eigen::Matrix<double,4,-1> & X)
{	
	//Ulrafast distance approximation, skip all the complex factoring
	// for(int i=0; i<A.cols(); i++)
	// {
	// 	X(0,i) = lott_distance_quadratic(F,A.col(i));
	// }
	// return;

	//Step 1: Compute the SVD of the upper left 2x2 of F
	// Eigen's SVD is also good, but a little slower.  This is provided for portability.
	const SVD2x2_Jacobi svd(F.block<2,2>(0,0));
	
	//Build the joint rotation of the 4D image space
	Eigen::Matrix<double,4,4> R;
	R.block<2,2>(0,0) =  svd.V().transpose();
	R.block<2,2>(2,0) = -svd.V().transpose();
	R.block<2,2>(0,2) =  svd.U().transpose();
	R.block<2,2>(2,2) =  svd.U().transpose();
	R *= M_SQRT1_2;

	//Step 2: Create the coefficient 4-vector that will become [c,d,e,f]
	const Eigen::Matrix<double,4,1> β(F(2,0), F(2,1), F(0,2), F(1,2));
	//Rotate the coefficients into the new blended joint image space
	const Eigen::Matrix<double,4,1> βr = R*β;

	Eigen::Matrix<double,4,1> Xod, X_hat;
	Eigen::Matrix<double,4,1> βrt;

	//For each point pair, triangulate the nearest valid point
	for(int i=0; i<A.cols(); i++)
	{
		//Step 3: Rotate point into the joint space: 
		// A = [u0,v0,u1,v1]^T
		const Eigen::Matrix<double,4,1> Ar = R*A.col(i); 

		//Step 4: Translate quadric (compute [c,d,e,f])
		double a = svd.d(0);
		double b = svd.d(1);

		βrt(0) = a*Ar(0) + βr(0);
		βrt(1) = b*Ar(1) + βr(1);
		βrt(2) = -a*Ar(2) + βr(2);
		βrt(3) = -b*Ar(3) + βr(3);  

		double & c = βrt(0);
		double & d = βrt(1);
		double & e = βrt(2);
		double & f = βrt(3);

		//Step 5: compute g
		double g = Ar.dot(βrt+βr) + 2.0*F(2,2);

		//Step 6: conditionally swap images if g is negative
		const bool swap_images = (g < 0);
		if(swap_images)
		{
			const double cp = -e;
			const double dp = -f;
			e = -c;
			f = -d;
			c = cp;
			d = dp;
			g = -g;
		}

		//Step 7: normalize by abs(c) has been replaced with a change of variables
		//        in the solution polynomial. p6(-c*x)/c^4.

		//**Guaranteed convergence to the root
		//Step 8: Build the polynomial and its derivative
		//Step 9: Converge to the solution using an itterative approach
		const double x = full_root_iterative(a,b,c,d,e,f,g);
		
		//** OR A variable approximation of the root can be faster though slightly less accurate
		// Select a performance speed versus accuracy 
		// from less to more accurate, faster to slower
		// const double x = householder_step_from_origin<3>(a,b,c,d,e,f,g);

		//Step 10: Assemble the point, and conditionally swap images back
		Xod(0) = x;
		Xod(1) = (d * x)/(c + (a-b) * x);
		Xod(2) = (e * x)/(c + (a+a) * x);
		Xod(3) = (f * x)/(c + (a+b) * x); 

		if(swap_images)// [z w x y] <- [x y z w]
		{
			const double y = Xod(1);
			Xod(0) = Xod(2);
			Xod(1) = Xod(3);
			Xod(2) = x;
			Xod(3) = y;
		}

		//Step 11: Transform back to original frame
		Xod += Ar;
		X_hat.head(2) = svd.V()*(Xod.head(2) - Xod.tail(2))*M_SQRT1_2;
		X_hat.tail(2) = svd.U()*(Xod.head(2) + Xod.tail(2))*M_SQRT1_2;
		//X_hat = [u0, v0, u1, v1]^T such that [u1,v1,1]*F*[u0,v0,1]^T = 0 exactly

		//Output
		X.col(i) = X_hat;
	}
}

