#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <qpOASES.hpp>
#include <vector>
#include<math.h>
#include<string>

#define PI 3.14159265

const int n_states = 103;
const float dt = 0.01;

using namespace Eigen;
using namespace std;

void loadAlift(Matrix<double, n_states, n_states> &AliftRef, string FileName) {
    ifstream AliftFile(FileName);
    if(!AliftFile.is_open()) cout << "ERROR: Alift File Open" << endl;

    int i = 0;
    int j = 0;
    for (int k = 0; k < n_states; k++) {
        string line;
        getline(AliftFile, line, '\n');
        if (i == 0) {
            line = line.substr(3, line.length()-1);
        }
        string entry;
        for (char c : line) {
            if (c == ';') {
                AliftRef(i, j) = stod(entry);
                j++;
                entry = "";
            } else {
                entry += c;
            }
        }
        AliftRef(i, j) = stod(entry);

        j = 0;
        i++;
    }
}

void loadBlift(Matrix<double, n_states, 1> &BliftRef, const string FileName) {
    ifstream BliftFile(FileName);
    if(!BliftFile.is_open()) cout << "ERROR: Blift File Open" << endl;
    
    int i = 0;
    for (int k = 0; k < n_states; k++) {
        string line;
        getline(BliftFile, line, '\n');
        if (i == 0) {
            line = line.substr(3, line.length());
        }
        BliftRef(i) = stod(line);
        i++;
    }
}

vector<int> NotNanIndex(MatrixXd &A, int n) {
    vector<int> indices;
    for (int i = 0; i < n; ++i) {
        if (!isnan(A(i, 0)))
            indices.push_back(i);
    }
    return indices;
}

void convertMatrixtoQpArray(qpOASES::real_t *qpArray, MatrixXd &Matrix, int m, int n) {
    int k = 0;
    for (int i = 0; i < m; ++i) {
        for (int j=0; j < n; ++j) {
            *(qpArray + k) = Matrix(i, j);
            ++k;
        }
    }
}

class Controller
{
    public:
        int Np, p;
        MatrixXd Ab;
        MatrixXd Xlb, Xub;
        MatrixXd Ulb, Uub;
        VectorXd ulin;
        MatrixXd M1, M2;
        SparseMatrix<double> C;
        double Q, d;
        qpOASES::SQProblem Qp;

        Controller(Matrix<double, n_states, n_states> &A, Matrix<double, n_states, 1> &B, SparseMatrix<double> &Cpar, double d_par, double Q_par, double R, double QN,
                   int N, MatrixXd &UlbP, MatrixXd &UubP, MatrixXd &XlbP, MatrixXd &XubP, VectorXd &ulinP, VectorXd &qlin, string solver = "qpoases") {

            Xlb = XlbP; Xub = XubP;             // Save state bounds as controller parameters
            Ulb = UlbP; Uub = UubP;             // Save control bounds as controller paramters
            C = Cpar;                           // Save output matrix as controller parameter
            d = d_par; Q = Q_par;               // Save wieght Q and affine term in the dynamics d as parameters
            ulin = ulinP;                       // Save linear term in the cost as controller parameter

            Np = N;                             // Number of Control Horizon points
            p = C.rows();                       // Number of outputs
            const int n = A.rows();             // Number of states
            const int m = B.cols();             // Number of control inputs

            VectorXd x0(n);                     // Dummy Variable

            // Handle state boundary matrices; convert matrices from n x 1 --> n x Np
            if (Xub.cols() == 1 || Xlb.cols() == 1) {
                if (Xub.size() != n || Xub.size() != n) {
                    cout << "The dimension of Xub or Xlb seems to be wrong" << endl;
                }
                VectorXd Xlb_temp = Xlb; Xlb = Xlb_temp.replicate(1, Np);
                VectorXd Xub_temp = Xub; Xub = Xub_temp.replicate(1, Np);
            }
            // Handle control input boundary matrices; convert matrices from m x 1 --> m x Np
            if (Uub.cols() == 1 || Ulb.cols() == 1) {
                if (Uub.size() != m || Ulb.size() != m) {
                    cout << "The dimension of Uub or Ulb seems to be wrong" << endl;
                }
                VectorXd Ulb_temp = Ulb; Ulb = Ulb_temp.replicate(1, Np);
                VectorXd Uub_temp = Uub; Uub = Uub.replicate(1, Np);
            }

            // Affine term in the dynamics - handled by state inflation (d=0)
            if (d != 0) {

                cout << "didn't think so lmao" << endl;
            } else {
                d = NAN;
            }

            // Linear term in the cost; convert matrix from m x 1 --> m*Np x 1
            if (ulin.size() == m) {
                ulin = ulin.reshaped();
                ulin = ulin.replicate(Np, 1);
            } else if (ulin.size() == Np*m) {
                ulin = ulin.reshaped();
            } else {
                ulin = MatrixXd::Constant(m*Np, 1, 0);
                cout << "Wrong size of ulin was input" << endl;
            }

            // Quadratic term in cost
            if (qlin.size() == p) {
                qlin = qlin.reshaped();
                qlin = qlin.replicate(Np, 1);
            } else if (qlin.size() == Np*p) {
                qlin = qlin.reshaped();
            } else {
                qlin = MatrixXd::Constant(p*Np, 1, 0);
                cout << "Wrong size of qlin was input" << endl;
            }

            // Create MPC matrices
            Ab = MatrixXd(Np*n, n); Ab(seq(0, n-1), seq(0, n-1)) = A;
            MatrixXd Bb(Np*n, Np*m); Bb(seq(0, n-1), seq(0, m-1)) = B;
            for (int i = 1; i < Np; ++i) {
                Ab(seq(i*n, (i+1)*n-1), seq(0,n-1)) = Ab(seq((i-1)*n, i*n-1), seq(0,n-1)) * A;
                Bb(seq(i*n, (i+1)*n-1), seq(0,Np*m-1)) = A * Bb(seq((i-1)*n, i*n-1), seq(0,Np*m-1));
                Bb(seq(i*n, (i+1)*n-1), seq(i*m, (i+1)*m-1)) = B;
            }      
           
            // Build the controller
            SparseMatrix<double> Qb(p*Np,p*Np);
            for (int i = 0; i < Np*p; ++i) {Qb.insert(i, i) = Q;}
            Qb.insert(p*Np-1, p*Np-1) = QN;

            vector<vector<double>> Cinserts;
            for (int k=0; k < C.outerSize(); ++k) {
                vector<double> Cinsert;
                for (SparseMatrix<double>::InnerIterator it(C,k); it; ++it) {
                    Cinsert.push_back(it.value()); Cinsert.push_back(it.row()); Cinsert.push_back(it.col()); 
                    Cinserts.push_back(Cinsert);
                    Cinsert.clear();
                }
            }
            SparseMatrix<double> Cb(Np*C.rows(), Np*C.cols());
            for (int i = 0; i < Np; ++i){
                for (int j = 0; j < Cinserts.size(); ++j) {
                    Cb.insert(p*i+Cinserts[j][1], n*i+Cinserts[j][2]) = Cinserts[j][0];
                }
            }

            SparseMatrix<double> Rb(Np,Np);
            for (int i = 0; i < Np; ++i) {Rb.insert(i, i) = R;}

            M1 = 2* ( ( Bb.transpose()* (Cb.transpose()*Qb*Cb) ) *Ab );
            M2 = (-2* (Qb*Cb) *Bb).transpose();

            // Bounds on the states
            MatrixXd Aineq_temp(n*Np*2, Np);
            MatrixXd bineq_temp(n*Np*2, 1);

            Xub = Xub.reshaped(); Xlb = Xlb.reshaped();
            Uub = Uub.reshaped(); Ulb = Ulb.reshaped();

            Aineq_temp(seq(0, Np*n-1), seq(0, Aineq_temp.cols()-1)) = Bb; Aineq_temp(seq(Np*n, 2*Np*n-1), seq(0, Aineq_temp.cols()-1)) = -Bb;
            bineq_temp(seq(0, Np*n-1), seq(0, bineq_temp.cols()-1)) = Xub - Ab*x0; bineq_temp(seq(Np*n, 2*Np*n-1), seq(0, bineq_temp.cols()-1)) = -Xlb + Ab*x0;

            vector<int> indices = NotNanIndex(bineq_temp, bineq_temp.rows());
            MatrixXd Aineq = Aineq_temp(indices, seq(0, Aineq_temp.cols()-1));
            MatrixXd bineq = bineq_temp(indices, seq(0, bineq_temp.cols()-1));

            MatrixXd H = 2*(Bb.transpose()*Cb.transpose()*Qb*Cb*Bb + Rb);
            MatrixXd g = (2*x0.transpose()*Ab.transpose()*Cb.transpose()*Qb*Cb*Bb).transpose() + ulin + Bb.transpose()*(Cb.transpose()*qlin);
            H = (H+H.transpose())/2;
            
            // Initialize controller with qpOASES
            int nV = Np;
            int nC = 2*Np;
            Qp = qpOASES::SQProblem(nV, nC);
            qpOASES::int_t nWSR = 100000;
            
            qpOASES::Options options;
	        Qp.setOptions( options );

            qpOASES::real_t H_qp[nV*nV]; convertMatrixtoQpArray(H_qp, H, nV, nV);
            qpOASES::real_t g_qp[nV]; convertMatrixtoQpArray(g_qp, g, nV, 1);
            qpOASES::real_t Aineq_qp[nC*nV]; convertMatrixtoQpArray(Aineq_qp, Aineq, nC, nV);
            qpOASES::real_t Ulb_qp[nV]; convertMatrixtoQpArray(Ulb_qp, Ulb, nV, 1);
            qpOASES::real_t Uub_qp[nV]; convertMatrixtoQpArray(Uub_qp, Uub, nV, 1);
            qpOASES::real_t bineq_qp[nC]; convertMatrixtoQpArray(bineq_qp, bineq, nC, 1);
            
        	// Solve first QP.
            Qp.init(H_qp, g_qp, Aineq_qp, Ulb_qp, Uub_qp, NULL, bineq_qp, nWSR);
            

        }


        double getOptVal(VectorXd x0, double yr)
        {
            return 0;
        }

};



int main()
{
    /* Importing System Dynamics Data */;
    const string FileAlift = "C:/Users/timme/OneDrive/Bureaublad/Tu Delft/DARE/Control Algorithm/Control Algorithm Tool/ControlSoftware/datafiles/Alift.csv";   // change to relative path
    const string FileBlift = "C:/Users/timme/OneDrive/Bureaublad/Tu Delft/DARE/Control Algorithm/Control Algorithm Tool/ControlSoftware/datafiles/Blift.csv";

    Matrix<double, n_states, n_states> Alift;
    Matrix<double, n_states, 1> Blift;

    loadAlift(Alift, FileAlift);
    loadBlift(Blift, FileBlift);

    /* Build reference signal */
    const float Tmax = 3;
    const int Nsim = Tmax/dt;
    float ymin, ymax;
    Vector2d x0;
    Matrix<double, 1, Nsim> yrr;
    int REF = 1; // Select type of reference signal (cos(1)/step(2))
    switch(REF) {
        case 1:
            ymin = -0.6;                                // constraint
            ymax = 0.6;                                 // constraint
            x0(0) = 0;                                  // initial conditions
            x0(1) = 0.6;
            for (int i = 0; i < Nsim; i++) {
                if (i < Nsim/2) {
                    yrr(i) = -0.3;
                } else {
                    yrr(i) = 0.3;
                }
            }
            break;
        case 2:
            ymin = -0.4;                                // constraint
            ymax = 0.4;                                 // constraint
            x0(0) = -0.1;                                  // initial conditions
            x0(1) = 0.1;
            for (int i = 0; i < Nsim; i++) {
                yrr(i) = (0.5*cos(2*PI*i/Nsim));
            }
            break;
    }

    /* Define Koopman controller */
    SparseMatrix<double> C(1, n_states); C.insert(0, 0) = 1;
    // Weight Matrices
    float Q = 1.0;
    float R = 0.01;
    // Prediction horizon
    float Tpred = 1.0;
    const int Np = Tpred/dt;
    // Constraints
    MatrixXd xlift_min = MatrixXd::Constant(n_states, 1, NAN); xlift_min(0) = ymin;
    MatrixXd xlift_max = MatrixXd::Constant(n_states, 1, NAN); xlift_max(0) = ymax;
    MatrixXd u_min(1, 1); u_min << -1;
    MatrixXd u_max(1, 1); u_max << 1;
    // Linear and quadratic terms in const function
    VectorXd ulin(1); ulin(0) = 0;
    VectorXd qlin(1); qlin(0) = 0;

    /* Build Koopman MPC Controller */

    Controller MPC(Alift, Blift, C, 0, Q, R, Q, Np, u_min, u_max, xlift_min, xlift_max, ulin, qlin);



    return 0;
}
