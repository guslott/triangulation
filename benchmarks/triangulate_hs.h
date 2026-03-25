
#pragma once
#include <vector>
#include <iostream>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/Polynomials>

#include "Polynomial.h"

Polynomial hartley_poly6(
	const double f, const double fp,
	const double a, const double b, const double c, const double d)
{
	Polynomial p(6);
	//Polynomial order (in the coefficients): [4,8,8,8,8,8,8]
	Eigen::Matrix<double,-1,1> & poly6 = p.coefficients();
	poly6(6) = b*d*(b*c - a*d); //t^0 term
	poly6(5) = (b*b*b*b + b*c*(b*c - a*d) + a*d*(b*c - a*d) + 2*b*b*d*d*fp*fp + d*d*d*d*fp*fp*fp*fp); //t^1 term
	poly6(4) = (4*a*b*b*b + a*c*(b*c - a*d) + 2*b*d*(b*c - a*d)*f*f + 4*b*b*c*d*fp*fp + 4*a*b*d*d*fp*fp + 4*c*d*d*d*fp*fp*fp*fp); //t^2 term
	poly6(3) = (6*a*a*b*b + 2*b*c*(b*c - a*d)*f*f + 2*a*d*(b*c - a*d)*f*f + 2*b*b*c*c*fp*fp + 8*a*b*c*d*fp*fp + 2*a*a*d*d*fp*fp + 6*c*c*d*d*fp*fp*fp*fp); //t^3 term
	poly6(2) = (4*a*a*a*b + 2*a*c*(b*c - a*d)*f*f + b*d*(b*c - a*d)*f*f*f*f + 4*a*b*c*c*fp*fp + 4*a*a*c*d*fp*fp + 4*c*c*c*d*fp*fp*fp*fp); //t^4 term
	poly6(1) = (a*a*a*a + b*c*(b*c - a*d)*f*f*f*f + a*d*(b*c - a*d)*f*f*f*f + 2*a*a*c*c*fp*fp + c*c*c*c*fp*fp*fp*fp); //t^5 term
	poly6(0) = a*c*(b*c - a*d)*f*f*f*f;
	return p;
}

double hartley_cost(
	const double t, 
	const double f, const double fp,
	const double a, const double b, const double c, const double d)
{
	const double denom1 = 1.0 + f*f*t*t;
	const double cost1 = (denom1 != 0.0) ? (t*t/denom1) : std::numeric_limits<double>::infinity();

	const double ct_d = c*t + d;
	const double at_b = a*t + b;
	const double denom2 = at_b*at_b + fp*fp*ct_d*ct_d;
	const double cost2 = (std::abs(denom2) > 1e-15) ? (ct_d*ct_d/denom2) : std::numeric_limits<double>::infinity();

	return cost1 + cost2;
}

void hartley_point_t(
	const double t,
	const double f, const double fp,
	const double a, double b, double c, double d,
	Eigen::Matrix<double,3,1> & x_est, 
	Eigen::Matrix<double,3,1> & xp_est)
{
	//Compute the pair of epipolar lines in the two images
	Eigen::Matrix<double,3,1> l, lp;
	l << t * f, 1, -t;
	lp << -fp*(c*t + d), a*t + b, c*t + d;

	//Closest point on line to the origin from equation provided by Hartley
	x_est(0) = -l(0)*l(2);
	x_est(1) = -l(1)*l(2);
	x_est(2) = l(0)*l(0) + l(1)*l(1);
	x_est /= x_est(2);

	xp_est(0) = -lp(0)*lp(2);
	xp_est(1) = -lp(1)*lp(2);
	xp_est(2) = lp(0)*lp(0) + lp(1)*lp(1);
	xp_est /= xp_est(2);
}

void hartley_triangulate(
	const Eigen::Matrix<double,3,-1> & x, 
	const Eigen::Matrix<double,3,-1> & xp,
	const Eigen::Matrix<double,3,3> & F,
	Eigen::Matrix<double,4,-1> & A)
{

	//HZ Pg 318:

	//SVD of entire 3x3 matrix to extract the epipoles
	Eigen::JacobiSVD<Eigen::Matrix<double,3,3> > svd( F, Eigen::ComputeFullV | Eigen::ComputeFullU );
	const Eigen::Matrix<double,3,3> U = svd.matrixU();
	const Eigen::Matrix<double,3,3> V = svd.matrixV();

	//Extract the epipoles
	const Eigen::Matrix<double,3,1> ep_ref = U.col(2);
	const Eigen::Matrix<double,3,1> e_ref = V.col(2).transpose();

	Eigen::Matrix<double,3,3> T = Eigen::Matrix<double,3,3>::Identity();
	Eigen::Matrix<double,3,3> Tp = Eigen::Matrix<double,3,3>::Identity();
	Eigen::Matrix<double,3,3> Ti = Eigen::Matrix<double,3,3>::Identity();
	Eigen::Matrix<double,3,3> Tpi = Eigen::Matrix<double,3,3>::Identity();

	for(int i=0; i<x.cols(); i++)
	{

		//step (i) Compute the translation
		T.block<2,1>(0,2) = -x.col(i).head(2);
		Tp.block<2,1>(0,2) = -xp.col(i).head(2);
		Ti.block<2,1>(0,2) = x.col(i).head(2);
		Tpi.block<2,1>(0,2) = xp.col(i).head(2);

		//Step (ii) - Translate the fundamental matrix
		Eigen::Matrix<double,3,3> Ft = Tpi.transpose()*F*Ti;

		//Step (iii) - compute the epipoles - translate from canonical
		Eigen::Matrix<double,3,1> e = T*e_ref;
		Eigen::Matrix<double,3,1> ep = Tp*ep_ref;
		
		//Normalize them so that e(0)*e(0) + e(1)*e(1) = 1
		e /= e.head(2).norm();
		ep /= ep.head(2).norm();

		//Step (iv)
		Eigen::Matrix<double,3,3> R, Rp;
		R << e(0), e(1), 0, -e(1), e(0),0, 0,0,1;
		Rp << ep(0), ep(1), 0, -ep(1), ep(0),0, 0,0,1;

		//Step (v)
		Eigen::Matrix<double,3,3> Frt = Rp*Ft*R.transpose();

		//Step (vi)
		double f = e(2);
		double fp = ep(2);
		double a = Frt(1,1);
		double b = Frt(1,2);
		double c = Frt(2,1);
		double d = Frt(2,2);

		//Step (vii) - form the polynomial according to eq 12.7
		Polynomial p = hartley_poly6(f,fp,a,b,c,d);

		std::vector<double> roots = p.real_roots_rpoly();
		if(roots.empty())
		{
			roots = p.real_roots_companion();
		}
		if(roots.empty())
		{
			roots = p.real_roots_sturm();
		}
		if(roots.empty())
		{
			roots.push_back(0.0);
		}


		//Step (viii) evaluate the cost function (eq 12.5) at each root
		double s_min = std::numeric_limits<double>::infinity();
		double t_min = 0.0;

		for(const double t : roots)
		{
			double s = hartley_cost(t,f,fp,a,b,c,d);
			if(s < s_min)
			{
				s_min = s;
				t_min = t;
			}
		}

		//Check against asymptotic value (t => inf)
		double s = std::numeric_limits<double>::infinity();
		const double denom_inf = a*a + fp*fp*c*c;
		if(std::abs(f) > 1e-12 && std::abs(denom_inf) > 1e-12)
		{
			s = 1.0/(f*f) + c*c/denom_inf;
		}
		if(s < s_min)
		{
			s_min = s;
			t_min = 1e10;
		}

		//Step (ix) - evaluate the parameterized line and solve for the image points
		//	they are the points on the parameterized line which is closest to the origin
		//	Note this will ALWAYS satisfy the fundamental matrix
		Eigen::Matrix<double,3,1> x_est,xp_est;
		hartley_point_t(t_min,f,fp,a,b,c,d,x_est,xp_est);

		//Step (x) - Transfer point pack into input space
		x_est = Ti*R.transpose()*x_est;
		xp_est = Tpi*Rp.transpose()*xp_est;

		A.col(i).head(2) = x_est.head(2) / x_est(2);
		A.col(i).tail(2) = xp_est.head(2) / xp_est(2);

	}
}
