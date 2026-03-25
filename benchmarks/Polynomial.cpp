/*
    Dr. Gus Lott
    guslott@yarcom.com
*/
#include "gkl/polynomials/Polynomial.h"

//Friends

std::ostream & 
operator << (
    std::ostream & out, 
    const gkl::polynomials::Polynomial & p)
{
    out << "[";
    for(uint32_t i=0; i<p.length()-1; i++)
    {
        out << p[i] << ", ";
    }
    out << p[p.length()-1] << "]";
    return out;
}

gkl::polynomials::Polynomial operator *(
    const double s,
    const gkl::polynomials::Polynomial & p)
{
    return (p*s);
}


//Constructors

gkl::polynomials::Polynomial::Polynomial() 
{ 
    p_.setZero(1,1); 
}

gkl::polynomials::Polynomial::Polynomial(
    const uint32_t order) 
{ 
    p_.setZero(order + 1,1); 
}


//Static Constructors
gkl::polynomials::Polynomial 
gkl::polynomials::Polynomial::from_coefficients(
    const Eigen::Matrix<double,-1,1> & c)
{
    if(c.rows()==0)
    {
        return Polynomial(0);
    }

    Polynomial p(c.rows()-1);
    for(uint32_t i = 0; i<c.rows(); i++)
    {
        p[i] = c(i);
    } 
    return p;
}

gkl::polynomials::Polynomial 
gkl::polynomials::Polynomial::from_roots(
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
gkl::polynomials::Polynomial::order() const
{
    return (p_.rows()-1);
}

uint32_t 
gkl::polynomials::Polynomial::degree() const
{
    return order();
}

uint32_t 
gkl::polynomials::Polynomial::length() const
{
    return p_.rows();
}


const Eigen::Matrix<double,-1,1> & 
gkl::polynomials::Polynomial::coefficients() const
{
    return p_;
}

Eigen::Matrix<double,-1,1> & 
gkl::polynomials::Polynomial::coefficients()
{
    return p_;
}

void 
gkl::polynomials::Polynomial::remove_leading_zeros()
{
    int zs = 0;
    Polynomial & p = *this;
    for(uint32_t i = 0; i<p.length(); i++)
    {
        if(p[i]==0)
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
gkl::polynomials::Polynomial::remove_roots_at_zero()
{
    uint32_t zs = 0;
    Polynomial & p = *this;
    for(uint32_t i = 0; i<p.length(); i++)
    {
        if(p[p.length()-1-i]==0)
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
gkl::polynomials::Polynomial::quadratic_polynomial_roots(
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
gkl::polynomials::Polynomial::linear_polynomial_root()
{
    const Polynomial & p = *this;
    assert(p.order() == 1);
    return -p[1]/p[0];
}

gkl::polynomials::Polynomial 
gkl::polynomials::Polynomial::divide_exact_by(
    const Polynomial & m) const
{
    const Polynomial & p = *this;
    assert(p.order() >= m.order());
    // If m is zero order, simple scalar divide
    // Otherwise, Levinson-Durbin recursion

    if(m.order() == 0)
    {
        //Simple scalar division
        return (p/m[0]);
    }

    //TODO: optional argument to hand in/out
    // the back vectors so we can skip all that factoring
    // when dividing by the same polynomial several times.
    // back vectors only depend upon m.

    //q = p/m where the division is known to be exact
    // minimizes roundoff error to improve precision
    // Solves for a q to minimize (M*q-p) = r
    // Use Levinson-Durbin Recursion
    // https://en.wikipedia.org/wiki/Levinson_recursion
    //Determine order of q:
    uint32_t q_order = p.order() - m.order();
    //Construct symmetric Toeplitz matrix and output vector, y
    // M*q = p
    // (M^T*M)*q = y = M^T*p
    // T*q = y

    //a is the first row of A = M^T*M up to the first zero coefficient
    Eigen::Matrix<double,1,-1> a = Eigen::Matrix<double,1,-1>::Zero(1,m.length());
    for(int j=0; j<m.length(); j++)
    {
        a(j) = m.coefficients().head(m.length() - j).dot(
                m.coefficients().tail(m.length() - j));
    }

    Eigen::Matrix<double,-1,1> y = Eigen::Matrix<double,-1,1>::Zero(q_order+1,1);
    for(int j=0; j<(q_order+1); j++)
    {
        for(int k=0; k<m.length(); k++)
        {
            y(j) += m[k]*p[j+k];
        }
    }

    // Eigen::Matrix<double,-1,-1> M = m.to_toeplitz(q_order+1);
    // Eigen::Matrix<double,-1,-1> T = M.transpose()*M;


    // Solves the following:
    // q.coefficients() = T.inverse() * y;
    // But does it in n^2 time vs n^3
    // Requires T is a symmetric Toeplitz matrix

    Polynomial q(0), q_pre(0);
    //T is a symmetric toeplitz matrix
    // bv is the back vector (ignore forward vectors since T is symmetric)
    Eigen::Matrix<double,-1,1> bv(1,1), bv_pre(1,1);
    bv(0) = 1.0 / a(0);
    q[0] = y(0)*bv(0);
    for(uint32_t i=1; i<y.rows(); i++)
    {
        bv_pre = bv;

        //Compute error term for back-vector
        uint32_t len_b = std::min(a.cols()-1,bv_pre.rows());
        double eps_b = a.block<1,-1>(0,1,1,len_b).dot(bv_pre.head(len_b));

        //Compute new back-vector
        bv = Eigen::Matrix<double,-1,1>::Zero(i+1,1);
        bv.tail(i) = bv_pre;
        bv -= Eigen::Matrix<double,-1,1>(eps_b * bv.reverse());
        bv /= (1.0 - eps_b*eps_b);

        //Update the solution using the back-vector
        q_pre = q;
        q = Polynomial(i);
        
        uint32_t len_n = std::min<uint32_t>(q_pre.length(),a.cols()-1);
        double eps_nm1 = a.reverse().block<1,-1>(0,a.cols()-len_n - 1,1,len_n).dot(
            q_pre.coefficients().tail(len_n));

        q.coefficients().head(i) = q_pre.coefficients();
        for(uint32_t k = 0; k<q.length(); k++)
        {
            q[k] += (y(i) - eps_nm1)*bv(k);
        }
    }
    return q;
}