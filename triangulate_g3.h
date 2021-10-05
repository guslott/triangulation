/*
	Fast Optimal L2 Triangulation by Direct Orthogonal Distance

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
#include <chrono>

//Note: None of the following functions accept DIM -1 (dynamic allocation), 
//but they also do not check for it.  They are templates here so the compiler
// can unwrap and embed them into the functions for fixed size vectors.

template <int DIM>
double polyval(
	const Eigen::Matrix<double,DIM,1> & p,
	const double x)
{
	if(x==0)
	{
		return p(DIM-1); //Return just the zero order term
	}
	//Evaulate polynomial using Horner's recursion:
	double px = p(0);
    for(uint32_t i=1; i < DIM; i++)
    {
        px = px * x + p(i); 
    }
    return px;	
}


//Attempt to hard code:
template <int DIM>
double converge_bisection(
	const Eigen::Matrix<double,DIM,1> & p, 
	std::pair<double,double> & bounds, 
	const double width_threshold,
	double & val_mid)
{
	double width = fabs(bounds.second - bounds.first);
    double midpoint = (bounds.second + bounds.first)/2.0;
    val_mid = polyval(p,midpoint);
    if(width < width_threshold)
    {
        return midpoint;
    }

    //Assumes we are guaranteed to have only one root in interval.
    // The sign on the left/right will be opposite
    // Keep it this way and bisect
    double val_left = polyval(p,bounds.first);
    double val_right = polyval(p,bounds.second);

    //Also should check that bounds.second >= bounds.first on input
    if(val_left*val_right > 0)
    {
        //Can't converge since the sign is the same on either side
        return midpoint;
    }

    for(int lps = 0; lps < 50; lps++)
    {
        if(width < width_threshold)
        {
            break;
        }

        if(val_mid*val_left > 0)
        {
            bounds.first = midpoint;
            val_left = val_mid;
        }else{
            bounds.second = midpoint;
            val_right = val_mid;
        }
        //Update values
        midpoint = (bounds.second + bounds.first)/2.0;
        width = bounds.second - bounds.first;
        val_mid = polyval(p,midpoint);
    }
    return midpoint;
}

template <int DIM>
double root_newton( //returns the root
    const double x, //The root to refine - return separate copy so it is not overwritten in an overshoot
    const Eigen::Matrix<double,DIM,1> & p,
    const Eigen::Matrix<double,(DIM-1),1> & dp,
    double & px,
    const uint32_t max_iterations,
    const double epsilon) //the polynomial value at the estimated root
{
	double root = x;
	px = polyval(p,root);
    for(int loops = 0; loops < max_iterations; loops++)
    {
        if(abs(px) < epsilon)
        {
            break;
        }
        const double dpx = polyval(dp,root);
        root -= (px/dpx);
        px = polyval(p,root);
    }
    return root;
}

template <int DIM>
double fine_convergence(
	const Eigen::Matrix<double,DIM,1> & p,
	const Eigen::Matrix<double,(DIM-1),1> & dp,
	std::pair<double,double> & bounds,
	const double val_thresh)
{
	//The following function is guaranteed to converge if
	//	- there is one real root in the bound
	//  - p(bound.first) * p(bound.second) < 0 (signs are different at the interval edges)

	//sort bounds:
	if(bounds.first > bounds.second)
	{
		double swap = bounds.first;
		bounds.first = bounds.second;
		bounds.second = swap;
	}

	//Specific to this problem, start at zero
	//Attempt an intial Newton-Raphson iteration (often succeeds very quickly - maybe always)
	//This is a Newton-Raphson descent from the query point to the surface of the quadric

	//TODO:  Collect Newton Method polynomial values to potentially accelerate a bisection if needed.
	// std::vector<std::pair<double,double> > p_samples;
	
	double px;//value of the polynomial - saves computing it again
	double root = root_newton(0,p,dp,px,20,val_thresh);

	if((abs(px) < val_thresh)&&(root>=bounds.first)&&(root<=bounds.second))
	{
		//p(root) is both small and within bounds, it is the root
		return root;
	}

	//Failed to converge in the aggressive approach on the large interval
	//Fine convergence on pathological polynomials
	double width_threshold = 1e-4;
	//Guaranteed slow convergence to a small window using bisection
	root = converge_bisection(p,bounds,width_threshold,px);

	//Final root polish using Newton-Raphson - Verify it doesn't diverge
	double tpx;
	double temp_root = root_newton(root,p,dp,tpx,5,val_thresh);
	//Test for improvement and staying within bounds
	if((abs(tpx) < abs(px)) && (root >= bounds.first) && (root <= bounds.second))
	{
		root = temp_root;
		px = tpx;
	}

	//Newton-Raphson has converged on the smaller interval
	if(abs(px) < val_thresh)
	{
		return root;
	}

	//Try a different method
	//TODO: Implement False Position method
	//TODO: Test convergence of this method

	//Pathological case, converge using bisection (guaranteed - slow)
	width_threshold = val_thresh;
	return converge_bisection(p,bounds,width_threshold,px);
}

void g3_poly6_x(
	const double a, const double b, 
	const double c, const double d, const double e, const double f, double g, 
	Eigen::Matrix<double,7,1> & p, Eigen::Matrix<double,6,1> & dp)
{
	//This enforces the constraint which should be maintained for a singular fundamental matrix
	// double g = (c*c - e*e)/a + (d*d - f*f)/b;
	//This is true of all fundamental matrices except when the epipole is at infinity in one image (b=0)
	const double nu2 = c*c + d*d + e*e + f*f;
	const double mu = a*a - b*b;
	const double lam = c*c - e*e;
	p(6) = a*b*c*c*c*c*g; //x^0 term
	p(5) = 2*a*b*c*c*c*(3*a*g+nu2);
	p(4) = b*c*c*(3*mu*lam+a*g*(13*a*a+b*b)+10*a*a*nu2);
	p(3) = 4*a*b*c*(2*mu*(c*c+lam)+a*g*(3*a*a+b*b)+4*a*a*nu2);
	p(2) = b*(mu*(a*a*(24*c*c+5*lam)-b*b*lam)+4*a*a*a*(g*(a*a+b*b)+8*a*nu2));
	p(1) = 4*a*b*c*mu*(5*a*a-b*b);
	p(0) = 4*a*a*b*mu*mu; //x^6 term

	//Hand out the derivative as well
	dp(5) = p(5)*1; //new x^0 term
	dp(4) = p(4)*2;
	dp(3) = p(3)*3;
	dp(2) = p(2)*4;
	dp(1) = p(1)*5;
	dp(0) = p(0)*6;
}

//This is the case for b=0 where the singularity constraint in poly6 above is false
// This is the 6th order version though it may be easily reduced to a simpler 
// fifth order version.  I left in one root at (c+ax) to keep it at order 6 for 
// simplicity of code.
void g3_poly6_b0_x(
	const double a, const double b,
	const double c, const double d, const double e, const double f, const double g,
	Eigen::Matrix<double,7,1> & p, Eigen::Matrix<double,6,1> & dp)
{
	p(6) = c*c*c*c*g; //x^0 term
	p(5) = 2*c*c*c*(c*c + d*d + e*e + f*f + 3*a*g);
	p(4) = a*c*c*(13*c*c + 10*d*d + 7*e*e + 10*f*f + 13*a*g);
	p(3) = 4*a*a*c*(8*c*c + 4*d*d + 2*e*e + 4*f*f + 3*a*g);
	p(2) = a*a*a*(37*c*c + 8*d*d + 3*e*e + 8*f*f + 4*a*g);
	p(1) = 20*a*a*a*a*c;
	p(0) = 4*a*a*a*a*a;

	//Hand out the derivative as well
	dp(5) = p(5)*1;
	dp(4) = p(4)*2;
	dp(3) = p(3)*3;
	dp(2) = p(2)*4;
	dp(1) = p(1)*5;
	dp(0) = p(0)*6;
}

/********* polynomial coefficients **************/
void g3_poly8_x(
	const double a, const double b, 
	const double c, const double d, const double e, const double f, double g, 
	Eigen::Matrix<double,9,1> & p, Eigen::Matrix<double,8,1> & dp)
{
	const double nu2 = c*c + d*d + e*e + f*f;
	const double gam = a*(c*c - e*e) + b*(d*d - f*f);
	p(8) = c*c*c*c*c*c*g;
	p(7) = 2*c*c*c*c*c*(nu2+4*a*g);
	p(6) = c*c*c*c*(3*gam+14*a*nu2+2*g*(13*a*a-b*b));
	p(5) = 2*c*c*c*(2*(a*a-b*b)*(c*c+e*e)+9*a*gam+19*a*a*nu2+a*g*(22*a*a-6*b*b));
	p(4) = c*c*(a*(a*a-b*b)*(25*c*c+15*e*e)+50*a*a*a*nu2+g*(41*a*a*a*a-26*a*a*b*b+b*b*b*b)+(39*a*a-b*b)*gam);
	p(3) = 2*c*((a*a-b*b)*(a*a*(29*c*c+9*e*e)-b*b*(c*c+e*e))+16*a*a*a*a*nu2+2*a*g*(5*a*a*a*a+b*b*b*b-6*a*a*b*b)+a*(18*a*a-2*b*b)*gam);
	p(2) = a*((a*a-b*b)*(a*a*(61*c*c+7*e*e)-3*b*b*(3*c*c+e*e))+8*a*a*a*a*nu2+4*a*g*(a-b)*(a-b)*(a+b)*(a+b)+(12*a*a*a-4*a*b*b)*gam);
	p(1) = 4*a*a*c*(a*a-b*b)*(7*a*a-3*b*b);
	p(0) = 4*a*a*a*(a*a-b*b)*(a*a-b*b);

	//Build Derivative
	dp(7) = p(7)*1; //new x^0 term
	dp(6) = p(6)*2;
	dp(5) = p(5)*3; 
	dp(4) = p(4)*4;
	dp(3) = p(3)*5;
	dp(2) = p(2)*6;
	dp(1) = p(1)*7;
	dp(0) = p(0)*8;
}

void g3_point_x(
	const bool swap_images,
	const double x, 
	const double a, const double b, 
	const double c, const double d, const double e, const double f,
	Eigen::Matrix<double,4,1> & Xod)
{
	Xod(0) = x;
	Xod(1) = d * x/(c + (a-b) * x);
	Xod(2) = e * x/(c + (a+a) * x);
	Xod(3) = f * x/(c + (a+b) * x);

	if(swap_images)
	{
		double x = Xod(0);
		double y = Xod(1);
		double z = Xod(2);
		double w = Xod(3);

		Xod(0) = z;
		Xod(1) = w;
		Xod(2) = x;
		Xod(3) = y;
	}
}

void g3_triangulate(
	const Eigen::Matrix<double,3,-1> & x, 
	const Eigen::Matrix<double,3,-1> & xp,
	const Eigen::Matrix<double,3,3> & F,
	Eigen::Matrix<double,4,-1> & X)
{
	
	//Factor the upper left 2x2 of the fundamental matrix
	Eigen::JacobiSVD<Eigen::Matrix<double,2,2> > svd(F.block<2,2>(0,0), Eigen::ComputeFullV | Eigen::ComputeFullU );
	Eigen::Matrix<double,2,2> U = svd.matrixU();
	Eigen::Matrix<double,2,2> V = svd.matrixV();
	Eigen::Matrix<double,1,2> d_ab = svd.singularValues();

	//Build the joint rotation of the 4D image space
	Eigen::Matrix<double,4,4> R;
	R.block<2,2>(0,0) =  V.transpose();
	R.block<2,2>(2,0) = -V.transpose();
	R.block<2,2>(0,2) =  U.transpose();
	R.block<2,2>(2,2) =  U.transpose();
	R /= M_SQRT2;


	//Create the coefficient 4-vector that will become [c,d,e,f]
	Eigen::Matrix<double,4,1> β;
	β << F(2,0), F(2,1), F(0,2), F(1,2);
	//Rotate the coefficients into the new blended joint image space
	const Eigen::Matrix<double,4,1> βr = R*β;

	//The polynomial and its derivative
	Eigen::Matrix<double,7,1> p;
	Eigen::Matrix<double,6,1> dp;

	Eigen::Matrix<double,4,1> Xod;
	Eigen::Matrix<double,4,1> A, Ar, βt;

	//For each point, triangulate the nearest valid point
	for(int i=0; i<x.cols(); i++)
	{
		//Rotate point into the diagonal fundamental matrix.
		//The joint image point in the original 4D image space
		A.head(2) = x.col(i).head(2)/x(2,i);
		A.tail(2) = xp.col(i).head(2)/xp(2,i);
		//The point rotated into the new space
		Ar = R*A;

		double a = d_ab(0);
		double b = d_ab(1);

		//Translate quadric (translation moves Ar to [0 0 0 0]^T - the origin)
		//ft = Dab * Ar + fr
		βt(0) = a*Ar(0) + βr(0);
		βt(1) = b*Ar(1) + βr(1);
		βt(2) = -a*Ar(2) + βr(2);
		βt(3) = -b*Ar(3) + βr(3);

		//Extract the polynomial coefficients
		double c = βt(0);
		double d = βt(1);
		double e = βt(2);
		double f = βt(3);
		double g = Ar.dot(βt+βr) + 2.0*F(2,2);

		//Now potentially rotate the problem to match the root bound inequality.
		// This places the origin "outside" of the quadric on the x-axis between infinity and the root
		// This assumes that a>0, b>=0.  This corresponds to an axis swapping rotation and 
		// an overall sign change of the coefficients to bring a/b back to positive.
		bool swap_images = (g < 0);
		if(swap_images)
		{
			//Must "swap images" 
			//	- trade x/y for z/w
			//	- Flip the sign
			double cp = -e;
			double dp = -f;
			double ep = -c;
			double fp = -d;
			double gp = -g;

			c = cp;
			d = dp;
			e = ep;
			f = fp;
			g = gp;
		}

		//Parameters Complete, prepare polynomial and converge using root bounds
		
		//Normalize all quadric coefficients
		double norm_factor = abs(c);
		a /= norm_factor;
		b /= norm_factor;
		c /= norm_factor;
		d /= norm_factor;
		e /= norm_factor;
		f /= norm_factor;
		g /= norm_factor;


		/* ***  Catch Special Case Motions *** */

		//Affine Camera case, a=b=0 - Coplanar principal planes
		if(abs(a)<1e-12)
		{
			double root = 0.5*c*g/(c*c + d*d + e*e + f*f);
			g3_point_x(swap_images,root,a,b,c,d,e,f,Xod);
			X.col(i) = (R.transpose() * (Xod + Ar));
			continue;
		}

		/* *** The following motion categories use the full polynomial *** */

		//b=0, motion where one camera is exactly in the principal plane of the other
		// but the principal planes are not parallel.
		if(abs(b)<1e-12)
		{
			//In this case, the "g-constraint" encoding singularity is no longer valid
			//If b==0, sixth order singularity constraint in g3_pol6_x(..) assumptions are invalid
			// The problem actually reduces to 5th order polynomial
			// Will keep it 6th order for simplicity here.  Left in a root at (c+ax), the epipoles
			g3_poly6_b0_x(a,b,c,d,e,f,g,p,dp);
		}else{

			//General case - direct sixth order polynomial
			//Do the full convergence to the point assumign b != 0
			//Computes the polynomial and its derivative
			// assumes g = (c*c - e*e)/a + (d*e - f*f)/b 
			//  - invalid for a = 0 or b = 0, these cases handled above
			g3_poly6_x(a,b,c,d,e,f,g,p,dp);
		}

		//Assemble the known root bounds
		std::pair<double,double> bounds;
		//the convergence function will sort this into the right order
		bounds.first = 0;
		bounds.second = -c/(a+a);

		//Reign in the infinite point bound for the case a=b where two roots appear at -c/(a+a)
		double omega = 1e6;
		double eps = abs(e)/(2*a*omega);
		bounds.second *= (1 - eps);

		//Converge to the single root within the bounds
		//Profile: 9/30/21 - This is still 75% of the execution time
		double root = fine_convergence(p,dp,bounds,1e-15);

		//Assemble the point
		//If the images were swapped, swap them back to the form expected
		g3_point_x(swap_images,root,a,b,c,d,e,f,Xod);
		
		//Transform back to original frame and output
		X.col(i) = (R.transpose() * (Xod + Ar));

	}
}





