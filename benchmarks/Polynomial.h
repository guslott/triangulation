/*
    Representation of a polynomial with real coefficients
    using Eigen C++ Library types

    Coefficients Stored in decreasing order:
    p(x) = 2*x^2 + 7*x + 4 
    Stores as:
    p.coefficients() << 2,7,4;
    p[0] = 2;

    TODO:
        - Clean up and separate the interface and implementation
        - Document interface

    Dr. Gus K. Lott
    guslott@gmail.com
*/

#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/Polynomials>

#include <cassert>
#include <complex>
#include <cmath>
#include <vector>
#include <iostream>

#include "rpoly.h"


class Polynomial
{
public:

    friend std::ostream & operator << (
        std::ostream & out, 
        const Polynomial & p);

    friend Polynomial operator * (
        const double s,
        const Polynomial & p);
    
    //Constructors
    //Zero'th order default: p(x) = 0 for all x
    Polynomial();
    //Zero polynomial with the specified order
    Polynomial(const uint32_t order);

    //Specify the coefficients in decreasing order
    static Polynomial from_coefficients(
        const Eigen::Matrix<double,-1,1> & c);
    
    //Specify the roots in any order (all real only)
    static Polynomial from_roots(
        const Eigen::Matrix<double,-1,1> & r);

    uint32_t order() const;
    uint32_t degree() const;
    uint32_t length() const;

    //Access const coefficients: p[x]
    double operator[](const uint32_t i) const
    {
        return p_(i);
    }

    //For editing coefficients
    double & operator[](const uint32_t i)
    {
        return p_(i);
    }

    const Eigen::Matrix<double,-1,1> & coefficients() const;

    //Bulk editing the coefficients
    Eigen::Matrix<double,-1,1> & coefficients();

    //Remove leading zeros so that p[0] != 0
    void remove_leading_zeros(const double abs_thresh = 0);

    //Returns the number of zero roots removed
    uint32_t remove_roots_at_zero(const double abs_thresh = 0);


    bool operator ==(
        const Polynomial & m)
    {
        Polynomial & p = *this;
        Polynomial q = p-m;
        if(q.coefficients().norm() == 0.0)
        {
            return true;
        }
        return false;
    }

    //Evaulate polynomial using Horner's recursion: p(x)
    double operator()(const double x) const
    {
        double px = p_(0);
        for(uint32_t i=1; i < p_.rows(); i++)
        {
            px = px * x + p_(i);
        }
        return px;
    }
    //Evaluate with a complex number
    std::complex<double> operator()(const std::complex<double> x) const
    {
        std::complex<double> px = p_(0);
        for(uint32_t i=1; i < p_.rows(); i++)
        {
            px = px * x + p_(i);
        }
        return px;
    }
    

    /***** Scalar Operations  **************************/

    Polynomial & operator+=(
        const double s)
    {
        Polynomial & p = *this;
        p[p.length()-1] += s;
        return *this;
    }
    Polynomial & operator-=(
        const double s)
    {
        Polynomial & p = *this;
        p[p.length()-1] -= s;
        return *this;
    }
    Polynomial & operator*=(
        const double s)
    {
        p_ *= s;
        return *this;
    }
    Polynomial & operator/=(
        const double s)
    {
        p_ /= s;
        return *this;
    }

    Polynomial operator+(
        const double s) const
    {
        Polynomial p = *this;
        p += s;
        return p;
    }
    Polynomial operator-(
        const double s) const
    {
        Polynomial p = *this;
        p -= s;
        return p;
    }
    Polynomial operator*(
        const double s) const
    {
        Polynomial p = *this;
        p *= s;
        return p;
    }
    Polynomial operator/(
        const double s) const
    {
        Polynomial p = *this;
        p /= s;
        return p;
    }

    /***** Polynomial Operations  **************************/

    Polynomial operator+(
        const Polynomial & m) const
    {
        Polynomial p = *this;
        p += m;   
        return p;
    }
    Polynomial operator-(
        Polynomial m) const
    {
        m *= -1;
        return (*this + m);
    }

    Polynomial operator*(
        const Polynomial & m) const
    {
        Polynomial p = *this;
        p *= m;
        return p;
    }

    Polynomial operator-() const
    {
        Polynomial p = *this;
        return p*(-1);
    }

    //This is overloading the divide operator assuming exact division is true
    Polynomial & operator/=(
        Polynomial & m)
    {
        *this = divide_exact_levinson(m);
        return *this;
    }
    Polynomial operator/(
        Polynomial m) const
    {
        Polynomial p = *this;
        p /= m;
        return p;
    }

    /*
        Polynomial addition
    */
    Polynomial & operator+=(
        const Polynomial & m)
    {
        //p = p + m
        Polynomial & p = *this;
        //May need to resize p_
        uint32_t max_order = p.order();
        bool resize = false;
        if(m.order() > p.order())
        {
            max_order = m.order();
            resize = true;
        }
        if(resize)
        {
            Eigen::Matrix<double,-1,1> c = p.coefficients();
            p_.setZero(max_order+1,1);
            p_.tail(c.rows()) = c;
        }

        p_.tail(m.length()) += m.coefficients();
        return p;
    }

    /*
        Polynomial subtraction (uses addition)
    */
    Polynomial & operator -=(
        Polynomial m)
    {
        m.p_ *= -1;
        return (*this += m);
    }

    /*
        Polynomial multiplication
    */
    Polynomial & operator*=(
        const Polynomial & m)
    {
        //q = p*m
        Polynomial & p = *this;
        Polynomial q(p.order() + m.order());

        for(uint32_t i = 0; i < p.length(); i++)
        {
            for(uint32_t j = 0; j < m.length(); j++)
            {
                q[i+j] += p[i]*m[j];
            }
        }
        *this = q;
        return *this;
    }

    void divide_by_linear_with_remainder(
        const double v, //divisor is m(x) = x + v
        Polynomial & q,
        double & a)
    {
        const Polynomial & p = *this;
        q.coefficients().resize(p.length()-1,1);
        if(v == 0)
        {
            q.coefficients() = p.coefficients().head(q.length());
            a = p[p.length()-1];
            return;
        }

        q[0] = p[0];
        for(uint32_t i = 1; i<q.length(); i++)
        {
            q[i] = p[i] - v*q[i-1];
        }
        a = p[p.length()-1] - v*q[q.length()-1];
    }

    //Formulation of a/b is as in Jenkins-Traub root finder
    Polynomial 
    divide_by_quadratic_with_remainder(
        Polynomial m,
        double & a,
        double & b)
    {
        Polynomial p = *this;
        assert(p.order() >= m.order());
        assert(m.order() == 2);

        p /= m[0];
        m /= m[0];
        double u = m[1];
        double v = m[2];

        Polynomial q(p.order() - 2);
        p.divide_by_quadratic_with_remainder(
            u,v,q,a,b);
        return q;
    }

    void divide_by_quadratic_with_remainder(
        const double u,
        const double v,
        Polynomial & q,
        double & a,
        double & b) const
    {
        const Polynomial & p = *this;
        q.coefficients().resize(p.length()-2,1);

        //"Generalization of the Horner recurrence to quadratic factors"
        // Factor a quadratic out of a polynomial as:
        // P(x) = Q(x)*M(x) + b*(x+u) + a;
        // M(x) = x^2 + u*x + v
        // This is a stable division of a polynomial by a quadratic factor,
        // m.  If the division is exact, b = a = 0.  Otherwise, 
        // b*(z+u)+a is the remainder of the division
        q[0] = p[0];
        if(q.order() > 0)
        {
            q[1] = p[1] - u*q[0];
            for(uint32_t i = 2; i < q.length(); i++)
            {
                q[i] = p[i] - u*q[i-1] - v*q[i-2];
            }
        }

        //For remainders:
        b = p[p.length()-2];
        b -= u*q[q.length()-1];
        if(q.order()>0)
        {
            b -= v*q[q.length()-2];
        }

        a = p[p.length()-1] - u*b;
        a -= v*q[q.length()-1];
    }

    //Group of functions for long division
    Polynomial
    long_division_quotient(
        const Polynomial & m) const
    {
        const Polynomial & p = *this;
        Polynomial q(p.order()-m.order());
        //Assemble quotient
        //Visualize this as back-substitution on a Toeplitz
        //Matrix M*q = p for the first q.length() rows of M
        for(int32_t i = 0; i<q.length(); i++)
        {
            q[i] = p[i];
            for(int32_t k = 1; (k < m.length()) && ((i-k)>=0); k++)
            {
                q[i] -= m[k]*q[i-k];
            }
            q[i] /= m[0];
        }
        return q;
    }

    Polynomial
    long_division_remainder(
        const Polynomial & m,
        const Polynomial & q) const
    {
        const Polynomial & p = *this;
        Polynomial r(m.order()-1);
        //Assemble remainder
        //Visualize this as back-substitution on the bottom 
        // m.length()-1 rows of M*q + r = p to yield r
        // r = Polynomial(m.order() - 1);
        for(int32_t i = 0; i<r.length(); i++)
        {
            r[i] = p[q.length()+i];

            for(int32_t k = 1+i; k<m.length(); k++)
            {
                int32_t qind = q.length() - (k - i);
                if(qind < 0)
                {
                    break;
                }
                r[i] -= m[k]*q[qind];
            }
        }
        return r;
    }

    void long_division(
        const Polynomial & m,
        Polynomial & q,
        Polynomial & r) const
    {
        //Polynomial p divided by m with remainder r
        //p = q*m + r, solve for q & r
        const Polynomial & p = *this;
        
        //If m is higher order than p, the quotient is zero
        // p = 0*m + r, or r = p
        if(p.order() < m.order())
        {
            q = Polynomial(0);
            r = p;
            return;
        }

        //Dividing by a scalar polynomial
        if(m.order() == 0)
        {
            q = p/m[0];
            r = Polynomial(0);
            return;
        }
        
        q = long_division_quotient(m);
        r = long_division_remainder(m,q);
    }

    /*
        Polynomial division where the divisor (m) is known
        to exactly divide into the polyomial
    */
    Polynomial 
    divide_exact_levinson(const Polynomial & m) const;

    /*
        Convert the polynomial coefficients to a toeplitz matrix for multiplication
        n_cols is the dimensionality of the coefficients of m where the
        matrix product, T(p)*m = q, is the polynomial product p*m = q
    */
    Eigen::Matrix<double,-1,-1> to_toeplitz(
        const int n_cols) const
    {
        const Polynomial & p = *this;

        int n_rows = p.order() + n_cols;
        Eigen::Matrix<double,-1,-1> T = Eigen::Matrix<double,-1,-1>::Zero(n_rows,n_cols);
        for (int i=0; i<n_cols; i++)
        {
            for(int j=0; j < p.length(); j++)
            {
                T(i+j,i) = p[j];
            }
        }
        return T;
    }

    /*
        Deriviative of the polynomial dP(x)/dx
    */
    Polynomial derivative() const
    {
        const Polynomial & p = *this;
        if(p.order() == 0)
        {
            return Polynomial(0);
        }
         
        Polynomial dp(p.order() - 1);
        dp.coefficients() = p.coefficients().head(p.length()-1);

        for(uint32_t i = 0; i < dp.length(); i++)
        {
            dp[i] *= double(p.order() - i);
        }
        return dp;
    }


    std::vector<std::complex<double> > roots() const;
    std::vector<double> real_roots_rpoly(const double imag_threshold = 1e-8) const;
    std::vector<double> real_roots_companion() const;
    std::vector<double> real_roots_sturm();
    

private:
    Eigen::Matrix<double,-1,1> p_;

    /*
        Iterative Convergence to a real root using
        Newton-Raphson iteration.  Initial value required
    */
    double root_newton(
        double rho,
        const uint32_t max_iterations = 100,
        const double epsilon = 1e-2) const
    {
        //TODO: Use a different root finding tolerance
        const Polynomial & p = *this;
        Polynomial dp = p.derivative();
        return root_newton(rho,dp,max_iterations,epsilon);
    }

    double root_newton(
        double rho,
        const Polynomial & dp,
        const uint32_t max_iterations = 100,
        const double epsilon = 1e-2) const
    {
        const Polynomial & p = *this;
        for(int loops = 0; loops < max_iterations; loops++)
        {
            double prho = p(rho);
            if(fabs(prho) < epsilon)
            {
                break;
            }
            double delta = prho/dp(rho);
            rho -= delta;
        }
        return rho;
    }

    /*
        Upper bound on the magnitude of the roots using Kojima's bound
        https://en.wikipedia.org/wiki/Properties_of_polynomial_roots#Other_bounds
    */
    double roots_kojima_upper_bound() const
    {
        const Polynomial & p = *this;

        double max_val = 0.0;
        for(uint32_t i=0; i<(p.length()-1); i++)
        {
            double c = fabs(p[i+1]/p[i]);
            if(i==(p.length() - 2))
            {
                c /= 2.0;
            }
            if(c > max_val)
            {
                max_val = c;
            }
        }
        return (max_val * 2.0);
    }

    double roots_cauchy_upper_bound() const
    {
        const Polynomial & p = *this;
        double max_coef = 0;
        for (int i = 1; i<p.length(); i++)
        {
            if(fabs(p[i])>max_coef)
            {
                max_coef = fabs(p[i]);
            }
        }
        return 1+max_coef/fabs(p[0]);
    }

    /*
        Cauchy lower bound for the absolute magnitude of the roots.
        Referenced in Jenkins-Traub real coefficient paper
    */
    double roots_cauchy_absolute_lower_bound() const
    {
        const Polynomial & p = *this;
        Polynomial m = p;
        m[0] = 1.0;
        for(uint32_t i=1; i<m.length(); i++)
        {
            m[i] = fabs(m[i]);
        }
        m[m.length()-1] *= -1;

        double rho = 1.0;
        rho = m.root_newton(rho);
        return rho;
    }

    double sign_at_positive_infinity() const
    {
        const Polynomial & p = *this;
        int z_ind = 0;
        //Find highest order non-zero coefficient
        for(int i=0; i<p.length(); i++)
        {
            if(p[i]!=0)
            {
                z_ind = i;
                break;
            }
        }
        //If is zero polynomial, return zero
        if(p[z_ind]==0)
        {
            return 0;
        }
        if(p[z_ind]>0)
        {
            return 1;
        }else{
            return -1;
        }
    }

    double sign_at_negative_infinity() const
    {
        const Polynomial & p = *this;
        int z_ind = 0;
        //Find highest order non-zero coefficient
        for(int i=0; i<p.length(); i++)
        {
            if(p[i]!=0)
            {
                z_ind = i;
                break;
            }
        }
        if(p[z_ind] == 0)
        {
            return 0;
        }
        //Find order of z_ind coefficient
        uint32_t ord = p.order() - z_ind;
        if((ord%2)==0)
        {
            //Even order
            if(p[z_ind] < 0)
            {
                return -1;
            }else{
                return 1;
            }

        }else{
            //odd order
            if(p[z_ind] < 0)
            {
                return 1;
            }else{
                return -1;
            }
        }

    }

    //Sturm Sequence
    //Useful for finding count/bounds on real roots of polynomials
    // https://en.wikipedia.org/wiki/Sturm%27s_theorem
    std::vector<Polynomial>
    sturm_sequence() const
    {
        const Polynomial & p = *this;

        std::vector<Polynomial> sturm_seq;

        sturm_seq.push_back(p);
        sturm_seq.push_back(p.derivative());

        for(int steps = 0; steps < p.order(); steps++)
        {
            const Polynomial & pi = sturm_seq[sturm_seq.size()-2];
            const Polynomial & pj = sturm_seq[sturm_seq.size()-1];
            Polynomial q,r;
            pi.long_division(pj,q,r);

            if((r.order() == 0) && (r[0] == 0))
            {
                break;
            }
            r.coefficients() *= -1;
            sturm_seq.push_back(r);
        }
        return sturm_seq;
    }

    //Using Sturm's Theorem and the signs at infinity, 
    // count total real roots of the polynomial
    uint32_t
    count_real_roots() const
    {
        const Polynomial & p = *this;
        std::vector<Polynomial> sturm_seq = p.sturm_sequence();

        uint32_t sign_changes_n = 0;
        uint32_t sign_changes_p = 0;
        double val_n = sturm_seq[0].sign_at_negative_infinity();
        double val_p = sturm_seq[0].sign_at_positive_infinity();
        
        for(uint32_t i = 1; i<sturm_seq.size(); i++)
        {
            double val = sturm_seq[i].sign_at_negative_infinity();
            if(val_n*val < 0)
            {
                sign_changes_n++;
            }
            val_n = val;

            val = sturm_seq[i].sign_at_positive_infinity();
            if(val_p*val < 0)
            {
                sign_changes_p++;
            }
            val_p = val;
        }
        return sign_changes_n - sign_changes_p;
    }

    void quadratic_polynomial_roots(
        std::complex<double> & r1,
        std::complex<double> & r2);

    double linear_polynomial_root();

    void 
    find_roots_in_bound(
        const std::vector<Polynomial> & sturm_seq,
        const double lower,
        const double upper,
        std::vector<double> & real_roots);
};







/* ************************************* ************************************* */
/* Methods for Polynomial                                                      */

//Friends

std::ostream & 
operator << (
    std::ostream & out, 
    const Polynomial & p)
{
    out << "[";
    for(uint32_t i=0; i<p.length()-1; i++)
    {
        out << p[i] << ", ";
    }
    out << p[p.length()-1] << "]";
    return out;
}

Polynomial operator *(
    const double s,
    const Polynomial & p)
{
    return (p*s);
}


//Constructors

Polynomial::Polynomial() 
{ 
    p_.setZero(1,1); 
}

Polynomial::Polynomial(
    const uint32_t order) 
{ 
    p_.setZero(order + 1,1); 
}


//Static Constructors
Polynomial 
Polynomial::from_coefficients(
    const Eigen::Matrix<double,-1,1> & c)
{
    if(c.rows()==0)
    {
        return Polynomial(0);
    }

    Polynomial p(c.rows()-1);
    p.coefficients() = c;
    return p;
}

Polynomial 
Polynomial::from_roots(
    const Eigen::Matrix<double,-1,1> & r)
{
    if(r.rows() == 0)
    {
        return Polynomial(0);
    }

    Polynomial p(0);
    p[0] = 1;

    //P is the product of monomials m(x) = (x-r_i)
    Polynomial m(1);
    m[0] = 1;
    for(uint32_t i=0; i<r.rows(); i++)
    {
        m[1] = -r(i);
        p = p * m;
    }
    return p;
}


//Accessors and size utilities

uint32_t 
Polynomial::order() const
{
    return (p_.rows()-1);
}

uint32_t 
Polynomial::degree() const
{
    return order();
}

uint32_t 
Polynomial::length() const
{
    return p_.rows();
}


const Eigen::Matrix<double,-1,1> & 
Polynomial::coefficients() const
{
    return p_;
}

Eigen::Matrix<double,-1,1> & 
Polynomial::coefficients()
{
    return p_;
}

void 
Polynomial::remove_leading_zeros(const double abs_thresh)
{
    int zs = 0;
    Polynomial & p = *this;
    for(uint32_t i = 0; i<p.length(); i++)
    {
        if(abs(p[i])<=abs_thresh)
        {
            zs++;
        }else{
            break;
        }
    }
    if(zs==0)
    {
        //Already tight
        return;
    }
    //Polynomial is all zeros
    if(zs == p.length())
    {
        p = Polynomial(0);
        return;
    }
    p_ = p_.tail(p.length()-zs).eval();
}

uint32_t
Polynomial::remove_roots_at_zero(const double abs_thresh)
{
    uint32_t zs = 0;
    Polynomial & p = *this;
    for(uint32_t i = 0; i<p.length(); i++)
    {
        if(fabs(p[p.length()-1-i])<=abs_thresh)
        {
            zs++;
        }else{
            break;
        }
    }
    if(zs==0)
    {
        return zs;
    }
    //Polynomial all zeros
    if(zs == p.length())
    {
        p = Polynomial(0);
        return zs;
    }
    p_ = p_.head(p.length()-zs).eval();

    return zs;
}

void 
Polynomial::quadratic_polynomial_roots(
    std::complex<double> & r1,
    std::complex<double> & r2)
{
    const Polynomial & p = *this;
    assert(p.order() == 2);
    //Factor quadratic
    //Compute discriminant, b^2 -4*a*c:
    double a = p[0];
    double b = p[1];
    double c = p[2];
    double disc = b*b - 4.0*a*c;
    if(disc < 0)
    {
        //Two Complex roots
        double r_re = -b/(2.0*a);
        double r_im = sqrt(fabs(disc))/(2.0*a);
        r1 = std::complex<double>(r_re,r_im);
        r2 = std::complex<double>(r_re,-r_im);
    }else{
        //Two real roots
        if(b >= 0)
        {
            r1 = (-b - sqrt(disc))/(2.0*a);
            r2 = (2.0*c) / (-b - sqrt(disc));
        }else{
            r1 = (-b + sqrt(disc))/(2.0*a);
            r2 = (2.0*c) / (-b + sqrt(disc));
        }
    }
}

double 
Polynomial::linear_polynomial_root()
{
    const Polynomial & p = *this;
    assert(p.order() == 1);
    return -p[1]/p[0];
}

Polynomial
Polynomial::divide_exact_levinson(
    const Polynomial & poly_m) const
{
    if(poly_m.order() == 0)
    {
        //Simple scalar division
        return (*this/poly_m[0]);
    }

    //  Treat the polynomial division problem as a least squares solution:
    //  Mq = p, which is a least squares problem when M is dimension p x q
    //  Solve: (M^T*M)q = (M^T*p) 
    //
    //  which is Tk * x = b
    //  Tk is a symmetric positive definite toeplitz matrix
    
    const Eigen::Matrix<double,-1,1> & p = this->coefficients();
    const Eigen::Matrix<double,-1,1> & m = poly_m.coefficients();
    
    //x will be the output polynomial coefficients
    Eigen::Matrix<double,-1,1> x = Eigen::Matrix<double,-1,1>::Zero(p.rows() - m.rows() + 1,1);

    //Compute the b vector (M^T * p)
    Eigen::Matrix<double,-1,1> b = Eigen::Matrix<double,-1,1>::Zero(x.rows(),1);
    for(int i=0; i<b.rows(); i++)
    {
        for(int j=0; j<m.rows(); j++)
        {
            b(i) += p(i+j) * m(j);
        }
    }
    
    //Compute the unique elements of the toeplitz matrix.  There are O(m) of these
    double r0 = m.squaredNorm();  //For scaling the diagonal to 1
    Eigen::Matrix<double,-1,1> r = Eigen::Matrix<double,-1,1>::Zero(b.rows()-1,1);
    for(int i=1; i < m.rows(); i++)
    {
        for(int j=i; j < m.rows(); j++)
        {
            r(i-1) += m(j-i)*m(j);
        }
    }
    //Normalize the equation so that the diagonal of Tk is 1
    r /= r0;
    b /= r0;

    int r_size = m.rows()-1; //the rest is zeros (the is the sparse component)


    //Levinson Algorithm for the General Right-Hand-Side Symmetric Toeplitz solve
    // Golub and van Loan (4th edition)
    // pg 211, Algorithm 4.7.2
    // Algorithm requires 4n^2 flops
    
    Eigen::Matrix<double,-1,1> y = Eigen::Matrix<double,-1,1>::Zero(b.rows()-1,1);
    Eigen::Matrix<double,-1,1> z = Eigen::Matrix<double,-1,1>::Zero(b.rows()-1,1);
    y(0) = -r(0);
    x(0) = b(0);
    double beta = 1;
    double alpha = -r(0);
    double mu;
    int n = r.rows();

    for(int k=0; k < n; k++)
    {
        beta = (1 - alpha*alpha)*beta;
        mu = b(k+1);
        for(int i=0; (i<=k)&&(i<r_size); i++)
        {
            mu -= r(i)*x(k-i);
        }
        mu /= beta;
        
        for(int i=0; i<=k; i++)
        {
            x(i) += mu * y(k-i);
        }
        x(k+1) = mu;

        if(k<(n-1))
        {
            //TODO: Note that these vectors (y) are only dependent on 
            // the divisor (m) and could be stored if many
            // polynomials are to be divided by m in a row
            alpha = -r(k+1);
            for(int i=0; (i<=k)&&(i<r_size); i++)
            {
                alpha -= r(i)*y(k-i);
            }
            alpha /= beta;

            for(int i=0; i<=k; i++)
            {
                z(i) = y(i) + alpha*y(k-i);
            }

            for(int i=0; i<=k; i++)
            {
                y(i) = z(i);
            }
            y(k+1) = alpha;
        }
    }

    return Polynomial::from_coefficients(x);
}


/* *************************************************************** */
/* Sturm Sequence root utilities                                   */

uint32_t sign_change_count(
    const std::vector<Polynomial> & sturm_seq,
    const double x)
{
    uint32_t sign_changes = 0;
    bool sign_pre = std::signbit(sturm_seq[0](x));

    for(uint32_t i = 1; i<sturm_seq.size(); i++)
    {
        bool sign_next = std::signbit(sturm_seq[i](x));

        if(sign_next^sign_pre)
        {
            sign_changes++;
        }
        sign_pre = sign_next;
    }
    return sign_changes;
}


//TODO: move these two bounded convergence functions to more global location (e.g. convergence.h)
double converge_false_position(
    const Polynomial & p,
    std::pair<double,double> & bounds,
    const double width_threshold = -1,
    const double height_threshold = -1)
{
    double a = bounds.first;
    double b = bounds.second;
    double fa = p(a);
    double fb = p(b);

    if((fa*fb) > 0)
    {
        //Can't converge, sign is same at ends of interval
        return (a+b)/2.0;
    }

    const double SMALL_ERR = 1e-7;
    if(std::abs(fa) < SMALL_ERR)
    {
        return a;
    }
    if(std::abs(fb) < SMALL_ERR)
    {
        return b;
    }

    double midpoint = (b+a)/2.0;
    double width = b - a;
    double height = fabs(fb - fa);

    double x = a;
    double fx = fa;
    double lfx = fa;

    //Illinois algorithm forcing convergence
    for(int lps = 0; lps < 100; lps++)
    {
        
        // std::cout << "lps: " << lps << " width: " << width << 
        // "  p(left): " << fa << " p(right): " << fb << std::endl;
        if(width < width_threshold)
        {
            break;
        }
        if(height < height_threshold)
        {
            break;
        }

        //Illinois Algorithm
        x = (fb * a - fa * b)/(fb - fa);
        fx = p(x);

        if(fabs(x) > SMALL_ERR)
        {
            if(fabs(fx/x) < SMALL_ERR)
            {
                break;
            }
        }else{
            if(fabs(fx) < SMALL_ERR)
            {
                break;
            }
        }

        if((fa*fx) < 0)
        {
            b = x;
            fb = fx;
            if((lfx*fx) > 0)
            {
                fa /= 2.0;
            }
        }else{
            a = x;
            fa = fx;
            if((lfx*fx)>0)
            {
                fb /= 2.0;
            }
        }

        lfx = fx;

        width = b - a;
        height = fabs(fb - fa);
        midpoint = (b+a)/2.0;
    }
    bounds.first = a;
    bounds.second = b;
    return x;
}

double converge_bisection(
    const Polynomial & p,
    std::pair<double,double> & bounds,
    const double width_threshold,
    const double height_threshold = -1)
{
    double width = fabs(bounds.second - bounds.first);
    double midpoint = (bounds.second + bounds.first)/2.0;
    if(width < width_threshold)
    {
        return midpoint;
    }

    //TODO: Convert to using std::signbit and xor for checking sign flips

    //Assumes we are guaranteed to have only one root in interval.
    // The sign on the left/right will be opposite
    // Keep it this way and bisect
    double val_left = p(bounds.first);
    double val_right = p(bounds.second);
    double height = fabs(val_right - val_left);

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
        if(height < height_threshold)
        {
            break;
        }
        double val_mid = p(midpoint);

        if(val_mid*val_left > 0)
        {
            bounds.first = midpoint;
            val_left = val_mid;
        }else{
            bounds.second = midpoint;
            val_right = val_mid;
        }
        
        midpoint = (bounds.second + bounds.first)/2.0;
        width = bounds.second - bounds.first;
        height = fabs(val_right - val_left);
    }
    
    return midpoint;
}

void 
Polynomial::find_roots_in_bound(
    const std::vector<Polynomial> & sturm_seq,
    const double lower,
    const double upper,
    std::vector<double> & real_roots)
{
    const Polynomial & p = sturm_seq[0];
    const Polynomial & dp = sturm_seq[1];

    uint32_t sig_lower = sign_change_count(sturm_seq,lower);
    uint32_t sig_upper = sign_change_count(sturm_seq,upper);
    uint32_t n_roots = sig_lower - sig_upper;
    
    std::vector<std::pair<double,double> > bounds(n_roots,std::pair<double,double>(lower,upper));

    //Coarse bound the roots to intervals containing one root
    //Shrink bounds just until each contains one root using Bisection
    uint32_t sig_left = sig_lower; 
    uint32_t sig_right = sig_upper;
    for(int i=0; i<bounds.size(); i++)
    {
        if(i!=0)
        {
            bounds[i].first = bounds[i-1].second;
            sig_left = sig_right;
            sig_right = sig_upper;
        }

        uint32_t count = sig_left - sig_right;

        for(int lps = 0; lps < 100; lps++)
        {
            if(count == 1)
            {
                break;
            }

            double midpoint = (bounds[i].second + bounds[i].first)/2.0;
            uint32_t sig_mid = sign_change_count(sturm_seq,midpoint);

            //If left count is zero, move left to this value
            uint32_t left_count = sig_left - sig_mid;
            if(left_count == 0)
            {
                bounds[i].first = midpoint;
                sig_left = sig_mid;
            }else{
                //else, move right to the value
                bounds[i].second = midpoint;
                sig_right = sig_mid;
            }
            count = sig_left - sig_right;
        }
    }//coarse bound loop

    //Fine convergence on each individual root
    for(int i=0; i<bounds.size(); i++)
    {
        //Bring bounds into a more reasonable range using bisection
        double width_threshold = 1e-4;
        double height_threshold = 1e-3;
        double root = converge_bisection(p,bounds[i],width_threshold,height_threshold);
        //Try Newton-Raphson, if fails, try false position, then bisection
        root = p.root_newton(root,dp,2,1e-15);
        
        //Did it converge?
        bool valid = true;
        if(p(root) > 1e-6)
        {
            valid = false;
        }else{
            if((root < bounds[i].first)|| (root > bounds[i].second))
            {
                valid = false;
            }
        }

        //Try False Position
        if(!valid)
        {
            root = converge_false_position(p,bounds[i],1e-3);
        }
        //Test if false position converged
        valid = true;
        if(p(root) > 1e-6)
        {
            valid = false;
        }else{
            if((root < bounds[i].first)|| (root > bounds[i].second))
            {
                valid = false;
            }
        }

        //Finally Try Bisection - Guaranteed, Slow
        if(!valid)
        {
            root = converge_bisection(p,bounds[i],1e-6,1e-6);    
        }
        
        //Refine using Netwon-Raphson on the final smaller bound
        root = p.root_newton(root,dp,20,1e-15);
        real_roots.push_back(root);
    }
}

std::vector<double> 
Polynomial::real_roots_rpoly(const double imag_threshold) const
{
    std::vector<double> real_roots;
    Polynomial p = *this;

    p.remove_leading_zeros();

    uint32_t roots_at_zero = p.remove_roots_at_zero();
    for(uint32_t i = 0; i < roots_at_zero; i++)
    {
        real_roots.push_back(0.0);
    }

    if(p.order() == 0)
    {
        return real_roots;
    }
    if(p.order() == 1)
    {
        real_roots.push_back(p.linear_polynomial_root());
        return real_roots;
    }
    if(p.order() == 2)
    {
        std::complex<double> r1,r2;
        p.quadratic_polynomial_roots(r1,r2);
        if(std::abs(r1.imag()) < imag_threshold)
        {
            real_roots.push_back(r1.real());
        }
        if(std::abs(r2.imag()) < imag_threshold)
        {
            real_roots.push_back(r2.real());
        }
        return real_roots;
    }

    uint32_t deg = p.order();
    std::vector<double> zeror(deg,0.0);
    std::vector<double> zeroi(deg,0.0);
    std::vector<double> coeffs(p.length(),0.0);

    for(uint32_t i = 0; i < p.length(); i++)
    {
        coeffs[i] = p[i];
    }

    int n_roots = rpoly(coeffs.data(), int(deg), zeror.data(), zeroi.data());
    if(n_roots < 0)
    {
        //Fall back to Sturm sequence finder if Jenkins-Traub fails
        std::vector<double> fallback = p.real_roots_sturm();
        real_roots.insert(real_roots.end(), fallback.begin(), fallback.end());
        return real_roots;
    }

    for(int i = 0; i < n_roots; i++)
    {
        if(std::abs(zeroi[i]) < imag_threshold)
        {
            real_roots.push_back(zeror[i]);
        }
    }

    //Fallback if no additional real roots were found
    if(real_roots.size() == roots_at_zero)
    {
        std::vector<double> fallback = p.real_roots_sturm();
        real_roots.insert(real_roots.end(), fallback.begin(), fallback.end());
    }

    return real_roots;
}

std::vector<double> 
Polynomial::real_roots_companion() const
{
    std::vector<double> real_roots;
    std::vector<std::complex<double> > all_roots = this->roots();

    for(int i=0; i<all_roots.size(); i++)
    {
        if(all_roots[i].imag()==0)
        {
            real_roots.push_back(all_roots[i].real());
        }
    }
    return real_roots;
}

std::vector<std::complex<double> > 
Polynomial::roots() const
{
    //Use the Companion Matrix Eigenvalue method
    //unsupported/Eigen/Polynomials
    int n = this->order() + 1;
    Eigen::Matrix<double,-1,1> q(n,1);
    //Requires flipped order from the matlab/numpy order otherwise used in this type
    for(int i=0; i<n; i++)
    {
        q((n-1)-i) = (*this)[i];
    }
    Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
    solver.compute(q);
    const Eigen::PolynomialSolver<double, Eigen::Dynamic>::RootsType &r = solver.roots();
    
    std::vector<std::complex<double> > roots;
    for(int i=0; i<r.size(); i++)
    {
        roots.push_back(r[i]);
    }
    return roots;
}

std::vector<double> 
Polynomial::real_roots_sturm()
{   
    std::vector<double> real_roots;

    //Clean polynomial of obvious issues
    remove_leading_zeros();

    uint32_t roots_at_zero = remove_roots_at_zero();
    for(uint32_t i=0; i < roots_at_zero; i++)
    {
        real_roots.push_back(0);
    }

    //Solve simple case polynomials
    if(order() == 0)
    {
        return real_roots;
    }
    if(order() == 1)
    {
        real_roots.push_back(linear_polynomial_root());
        return real_roots;
    }
    if(order() == 2)
    {
        std::complex<double> r1,r2;
        quadratic_polynomial_roots(r1,r2);
        if(r1.imag() == 0)
        {
            real_roots.push_back(r1.real());
            real_roots.push_back(r2.real());
        }
        return real_roots;
    }
    
    //order() > 2, Compute Sturm Sequence and bound roots
    std::vector<Polynomial> sturm_seq = sturm_sequence();
    
    //Bound the roots
    double absolute_upper_bound = roots_kojima_upper_bound();

    //Find positive roots
    find_roots_in_bound(sturm_seq,0,absolute_upper_bound,real_roots);
    //Find negative roots
    find_roots_in_bound(sturm_seq,-absolute_upper_bound,0,real_roots);

    return real_roots;
}
