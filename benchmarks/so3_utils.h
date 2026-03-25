#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/Polynomials>

Eigen::Matrix<double,3,3>
skew(const Eigen::Matrix<double,3,1> & phi)
{
    Eigen::Matrix<double,3,3> Px;
    Px << 0,-phi(2), phi(1), 
          phi(2), 0, -phi(0),
          -phi(1), phi(0), 0;
    return Px;
}

Eigen::Matrix<double,3,3>
so3_exp(const Eigen::Matrix<double,3,1> & phi)
{
	Eigen::Matrix<double,3,3> R = Eigen::Matrix<double,3,3>::Identity();
	double theta = phi.norm();
	if(theta == 0){
		return R;
	}

	//Rodrigues formula, not great for small angles
	Eigen::Matrix<double,3,3> S = skew(phi);
	R += S * (sin(theta)/theta) + S * S * (1-cos(theta))/(theta*theta);

	return R;
}

Eigen::Matrix<double,3,3>
rand_rotation_deg(const double angle_deg)
{
	Eigen::Matrix<double,3,1> phi = Eigen::Matrix<double,3,1>::Random();
	phi *= angle_deg*M_PI/180.0/phi.norm();
	return so3_exp(phi);
}

double gaussian_noise(double sigma){
	
    // Box-Muller transform to generate Gaussian random variable
    double u1 = Eigen::internal::random<double>(0, 1);
    double u2 = Eigen::internal::random<double>(0, 1);
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
	//the second value is uncorrelated, but we don't need it
	//double z2 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
	
    return sigma * z;
};

Eigen::Matrix<double,2,1> gaussian_noise2d(double sigma){
	
    // Box-Muller transform to generate Gaussian random variable
    double u1 = Eigen::internal::random<double>(0, 1);
    double u2 = Eigen::internal::random<double>(0, 1);
	Eigen::Matrix<double,2,1> noise;
	noise(0) = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
	noise(1) = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
	
    return sigma * noise;
};
