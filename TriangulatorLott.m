classdef TriangulatorLott
    %Triangulator Lott
    
    methods (Static)
        
        function eps2 = approx2_reprojection_error(F,x0,x1)
            %Compute the approximate squared reprojection error
            % Quadratic approximation using Newton step
            r1 = (F(1,1) + F(2,2))^2 + (F(1,2) - F(2,1))^2;
            r2 = (F(1,1) - F(2,2))^2 + (F(1,2) + F(2,1))^2;
            
            a = 0.5*(sqrt(r1) + sqrt(r2));
            nu2 = (F(1,:)*x0)^2 + (F(2,:)*x0)^2 + (x1'*F(:,1))^2 + (x1'*F(:,2))^2;
            g = 2*x1'*F*x0;
            
            eps2 = g^2 * nu2 / (6*a*g + 2*nu2)^2;
        end
        
        function [X,a,b,c,d,e,f,g] = triangulate(F,x0,x1)
            %triangulate - compute optimal point in image space
            % object initialized with fundamental matrix
            % input is 2x1 or 3x1 point vectors.  Output is 4x1 joint image
            % point.
            
            %Step 1: Compute the SVD (params: a & b), build 4x4 Rotation
            [U,D,V] = svd(F(1:2,1:2));
            
            d_ab = diag(D);
            R = [V',U';
                -V',U']/sqrt(2);
            
            %Step 2: Rotate reference parameters
            beta_r = R * [F(3,1), F(3,2), F(1,3), F(2,3)]';
            
            %Compute the point-dependent quadric parameters
            %Step 3: Rotate the measured image point off the surface
            A(1:2,1) = x0(1:2);
            A(3:4,1) = x1(1:2);
            Ar = R*A;
            
            a = d_ab(1);
            b = d_ab(2);
            %Step 4: Translate the imaged point to the origin
            beta_t = diag([a,b,-a,-b])*Ar + beta_r;
            
            c = beta_t(1);
            d = beta_t(2);
            e = beta_t(3);
            f = beta_t(4);
            
            %Step 5: compute g
            g = Ar'*(beta_t+beta_r) +2*F(3,3);
            
            %Step 6: Swap images if g is negative
            swap_images = (g<0);
            if(swap_images)
                cp = -e;
                dp = -f;
                ep = -c;
                fp = -d;
                gp = -g;
                
                c = cp; d = dp; e = ep; f = fp; g = gp;
            end
            
            %Step 7: normalize to abs(c).  Replaced by change of variables
            %in the polynomial: p6(-c*x)/c^4
            
            %Step 8 & 9: Build polynomial and converge from zero
            x = TriangulatorLott.full_root_iterative(a,b,c,d,e,f,g);
            
            %Step 10: compute 4D point and conditionally swap images back
            X0(1,1) = x;
            X0(2,1) = d*x./(c + (a-b)*x);
            X0(3,1) = e*x./(c + (a+a)*x);
            X0(4,1) = f*x./(c + (a+b)*x);
            
            if(swap_images)
                x = X0(1);
                y = X0(2);
                z = X0(3);
                w = X0(4);
                
                X0(1) = z;
                X0(2) = w;
                X0(3) = x;
                X0(4) = y;
            end
            
            %Step 11: transform point pair back to original image frame
            X = R'*(X0(1:4) + Ar);
        end
    
        function x = full_root_iterative(a,b,c,d,e,f,g)
            
            %Assemble the full polynomial
            [p,dp] = TriangulatorLott.poly6_mcx(a,b,c,d,e,f,g);
            
            x = TriangulatorLott.householder_step(p,4); %Initialize from zero
            
            for i=1:15
                px = polyval(p,x);
                if abs(px)<1e-14
                    break;
                end
                dpx = polyval(dp,x);
                x = x - px/dpx;
            end
            
            x = -c*x; %Scaling due to change of variables in polynomial
        end
        
        function [p,dp] = poly6_mcx(a,b,c,d,e,f,g)
            %The orthogonal distance polynomial.  p6(-c*x)/c^4 (see Table 1
            %in paper).  Instead of normalizing to abs(c), changed
            %variables to condition near the origin.
            
            nu2 = c*c + d*d + e*e + f*f;
            rho = a*(c*c - e*e) + b*(d*d - f*f);
            
            p(7) = g; %//x^0 term
            p(6) = -(6*a*g + 2*nu2);
            p(5) = (3*rho + g*(13*a*a - 2*b*b) + 10*a*nu2);
            p(4) = -(8*(a*a - b*b)*(2*c*c - e*e) + 4*a*g*(3*a*a + b*b) + 16*a*a*nu2);
            p(3) = ((a*a - b*b)*(a*(29*c*c - 5*e*e)+b*(d*d - f*f)) + g*(4*a*a*a*a + 3*a*a*b*b + b*b*b*b) + 8*a*a*a*nu2);
            p(2) = -4*c*c*(a*a - b*b)*(5*a*a - b*b);
            p(1) = 4*c*c*a*(a*a - b*b)*(a*a - b*b);

            %//Hand out the derivative as well
            dp(6) = p(6)*1; %//new x^0 term
            dp(5) = p(5)*2;
            dp(4) = p(4)*3;
            dp(3) = p(3)*4;
            dp(2) = p(2)*5;
            dp(1) = p(1)*6;
        end
        
        function xh = householder_step(p,order)
            
            k0 = p(7);
            k1 = p(6);
            k2 = p(5);
            k3 = p(4);
            k4 = p(3);
            k5 = p(2);
            %k6 = p(1);
            
            switch order
                case 1
                    %Newton-Raphson's Method
                    num = -k0;
                    den =  k1;
                case 2
                    %Halley's method
                    num = -k0*k1;
                    den = (k1*k1 - k0*k2);
                case 3
                    num = -(k0*k1*k1 - k0*k0*k2);
                    den =  (k1*k1*k1 - 2*k0*k1*k2 + k0*k0*k3);
                case 4
                    num = -(k0*k1*k1*k1 - 2*k0*k0*k1*k2 + k0*k0*k0*k3);
                    den =  (k1*k1*k1*k1 - 3*k0*k1*k1*k2 + k0*k0*k2*k2 + 2*k0*k0*k1*k3 - k0*k0*k0*k4);
                case 5
                    num = -k0*k1*k1*k1*k1 + 3*k0*k0*k1*k1*k2 -   k0*k0*k0*k2*k2 - 2*k0*k0*k0*k1*k3 + k0*k0*k0*k0*k4;
                    den =  k1*k1*k1*k1*k1 - 4*k0*k1*k1*k1*k2 + 3*k0*k0*k1*k2*k2 - 2*k0*k0*k0*k1*k4 + k0*k0*k0*k0*k5 + 3*k0*k0*k1*k1*k3 - 2*k0*k0*k0*k2*k3;
                otherwise
                    error('TriangulatorLott: Invalid Householder Order (not implemented)');
                    
            end
            xh = num/den;
        end
        
    end
end

