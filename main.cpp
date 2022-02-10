#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <qpOASES.hpp>
#include <vector>
#include <math.h>
#include <string>

#define PI 3.14159265

const int n_states = 103;
const float dt = 0.01;

using namespace Eigen;
using namespace std;

MatrixXd loadFile(string FileName, int row, int col) {
    MatrixXd Matrix(row, col);
	ifstream File(FileName);
	for (int k = 0; k < row; ++k) {
		string line;
		getline(File, line, '\n');
		for (int j = 0; j < col; ++j) {
			string entry = line.substr(0, line.find(','));
			line.erase(0, line.find(',') + 1);
			Matrix(k, j) = stod(entry);
		}
	}
    return Matrix;
}

void saveToFile(MatrixXd &data, int rows, int cols, string FileName) {
    ofstream File; File.open(FileName);

    for (int k = 0; k < rows; ++k) {
        string line = "";
        for (int j = 0; j < cols; ++j) {
            line.append(to_string(data(k, j)));
            line.append(",");
        }
        line.pop_back();
        File << line << "\n";
    }
    File.close();
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

class rbf {
    public:
        MatrixXd C;
        string rbf_type;
        int eps, k, Nrbf;

        rbf(MatrixXd &cent, string rbf_typeP, int epsP = 1, int kP = 1) {
            // Initialize the lifting function
            C = cent;
            Nrbf = cent.cols();
            rbf_type = rbf_typeP;
            eps = epsP; k = kP;
        }

        VectorXd liftState(VectorXd &x) {
            // Create lifted matrix
            int Nstate = x.rows();
            VectorXd Y(Nrbf + Nstate, 1);
            
            // Populate lifted data matrix
            for (int i = Nstate; i < Nrbf + Nstate; ++i) {
                MatrixXd Cstate = C(seq(0, Nstate-1), i-3);
                double r_squared = (x-Cstate).dot(x-Cstate);
                double y;
            
                if (rbf_type == "thinplate") {
                    y = r_squared*log(sqrt(r_squared));
                } else if (rbf_type == "gauss") {
                    y = exp(- pow(eps, 2) * r_squared);
                } else if (rbf_type == "invquad") {
                    y = 1 / (1 + pow(eps, 2) * r_squared);
                } else if (rbf_type == "invmultquad") {
                    y = 1 / sqrt(1 + pow(eps,2) * r_squared);
                } else if (rbf_type == "polyharmonic") {
                    y = pow(r_squared, k/2) * log(sqrt(r_squared));
                } else {
                    cout << "RBF type not recognized";
                } 
              
                if (y == NAN) {
                    y = 0;
                }

                Y(i) = y;
            }
            Y(seq(0, Nstate-1)) = x;
            
            return Y;
        }
};

class dynamics {
    private:
        VectorXd k1;
        VectorXd k2;
        VectorXd k3;
        VectorXd k4;
        VectorXd stateDerivative;
        VectorXd stateEv;
        void DyDt(double u) {
            /*
            t - 1-D array representing the independent variable of the DE
            y - N-D array representing the state variable of the DE
            */
            stateDerivative[0] = 19.10828025-39.3153*stateEv[0]-32.2293*stateEv[1]*u;
            stateDerivative[1] = -3.333333333-1.6599*stateEv[1]+22.9478*stateEv[0]*u;
        }
    public:
        double dt;
        double t;
        int n_states;
        VectorXd state;
        dynamics(int n, VectorXd initState = VectorXd::Zero(2,1), double dtP = 0.01, double tP = 0) {
            // Initialize system properties
            dt = dtP;
            t = tP;
            n_states = n;
            state = initState;

            // Initialize private data members
            k1 = VectorXd::Zero(n,1);
            k2 = VectorXd::Zero(n,1);
            k3 = VectorXd::Zero(n,1);
            k4 = VectorXd::Zero(n,1);
            stateDerivative = VectorXd::Zero(n,1);
        }
        
        void updateState(double u){
            // Evaluation at start of interval
            stateEv = state;
            DyDt(u);
            k1 = stateDerivative;

            // Evaluation at midway of interval
            t + 1/2*dt;
            stateEv = state + dt*k1/2.0;
            DyDt(u);
            k2 = stateDerivative;

            stateEv = state + dt*k2/2.0;
            DyDt(u);
            k3 = stateDerivative;

            // Evaluation at end of interval
            t + 1/2*dt;
            stateEv = state + dt*k3;
            DyDt(u);
            k4 = stateDerivative;

            state = state + dt*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
        }
};

class Controller
{
    private:
        qpOASES::QProblem Qp; 
        qpOASES::real_t H_qp[100*100];
    public:
        int Np, p;
        MatrixXd Ab;
        MatrixXd Xlb, Xub;
        MatrixXd Ulb, Uub;
        VectorXd ulin;
        MatrixXd M1, M2;
        SparseMatrix<double> C;
        double Q, d;

        Controller(MatrixXd &A, MatrixXd &B, SparseMatrix<double> &Cpar, double d_par, double Q_par, double R, double QN,
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
                if (Xub.size() != n || Xlb.size() != n) {
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
                VectorXd Uub_temp = Uub; Uub = Uub_temp.replicate(1, Np);
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
            Qp = qpOASES::QProblem(nV, nC);
            qpOASES::int_t nWSR = 1000;
            
            qpOASES::Options options;
            options.setToMPC();
            options.printLevel = qpOASES::PL_LOW;
	        Qp.setOptions( options );

            convertMatrixtoQpArray(H_qp, H, nV, nV); 
            qpOASES::real_t g_qp[nV]; convertMatrixtoQpArray(g_qp, g, nV, 1);
            qpOASES::real_t Aineq_qp[nC*nV]; convertMatrixtoQpArray(Aineq_qp, Aineq, nC, nV);
            qpOASES::real_t Ulb_qp[nV]; convertMatrixtoQpArray(Ulb_qp, Ulb, nV, 1);
            qpOASES::real_t Uub_qp[nV]; convertMatrixtoQpArray(Uub_qp, Uub, nV, 1);
            qpOASES::real_t bineq_qp[nC]; convertMatrixtoQpArray(bineq_qp, bineq, nC, 1);
            
            qpOASES::SymDenseMat *Hsd = new qpOASES::SymDenseMat(100, 100, 100, H_qp);
	        qpOASES::DenseMatrix *Ad = new qpOASES::DenseMatrix(200, 100, 100, Aineq_qp);

            // Solve first QP.
            Qp.init(Hsd, g_qp, Ad, Ulb_qp, Uub_qp, NULL, bineq_qp, nWSR);
        }


        double getOptVal(VectorXd x0, double yrr)
        {
            // Reference state and current state
            VectorXd yr = VectorXd::Constant(Np, yrr);

            if (!isnan(d)) {
                x0 = x0;
            }

            // Linear part of constraints
            MatrixXd bineq_temp(n_states*Np*2, 1);
            bineq_temp(seq(0, Np*n_states-1), seq(0, bineq_temp.cols()-1)) = Xub - Ab*x0; bineq_temp(seq(Np*n_states, 2*Np*n_states-1), seq(0, bineq_temp.cols()-1)) = -Xlb + Ab*x0;
            vector<int> indices = NotNanIndex(bineq_temp, bineq_temp.rows());
            MatrixXd bineq = bineq_temp(indices, seq(0, bineq_temp.cols()-1));

            // Linear part of the objective function
            MatrixXd g = M1*x0 + M2*yr + ulin;

            // Solve Qp
            int nV = Np;
            int nC = 2*Np;
            qpOASES::int_t nWSR = 1000;
            qpOASES::real_t U[nV];

            qpOASES::real_t g_qp[nV]; convertMatrixtoQpArray(g_qp, g, nV, 1);
            qpOASES::real_t Ulb_qp[nV]; convertMatrixtoQpArray(Ulb_qp, Ulb, nV, 1);
            qpOASES::real_t Uub_qp[nV]; convertMatrixtoQpArray(Uub_qp, Uub, nV, 1);
            qpOASES::real_t bineq_qp[nC]; convertMatrixtoQpArray(bineq_qp, bineq, nC, 1);

            Qp.hotstart(g_qp, Ulb_qp, Uub_qp, NULL, bineq_qp, nWSR);
            
            Qp.getPrimalSolution(U);                                                            // get optimal control inputs over event horizon
            MatrixXd y = C*x0;
            double optval = Qp.getObjVal() + (y.transpose()*Q*y)(0);
            
            return U[0];
        }
};


int main()
{
    /* Dynamics Properties */
    int n = 2;                              // number of states
    int m = 1;                              // number of control inputs
    Matrix<double, 1, 2> Cy {0, 1};         // output matrix: y = Cy*x
    int nD = 1;                             // number of delays
    int ny = Cy.rows();                     // number of outputs
    int n_zeta = (nD+1)*ny + nD*m;          // dimension of delay-embedded state

    dynamics Motor(n, VectorXd::Zero(2,1), dt, 0);
    dynamics MotorNom(n, VectorXd::Zero(2,1), dt, 0);

    /* Importing System Dynamics Data */;
    const string FileAlift = "C:/Users/timme/OneDrive/Bureaublad/Tu Delft/DARE/Control Algorithm/Control Algorithm Tool/ControlSoftware/datafiles/Alift.csv";   // change to relative path
    const string FileBlift = "C:/Users/timme/OneDrive/Bureaublad/Tu Delft/DARE/Control Algorithm/Control Algorithm Tool/ControlSoftware/datafiles/Blift.csv";
    const string FileCent = "C:/Users/timme/OneDrive/Bureaublad/Tu Delft/DARE/Control Algorithm/Control Algorithm Tool/ControlSoftware/datafiles/cent.csv";

    MatrixXd Alift = loadFile(FileAlift, n_states, n_states);
    MatrixXd Blift = loadFile(FileBlift, n_states, 1);

    /* Configure lifting function */
    int Nrbf = 100;
    //MatrixXd cent = MatrixXd::Random(n_zeta, Nrbf);
    MatrixXd cent = loadFile(FileCent, n_zeta, Nrbf);
    string rbf_type = "thinplate";
    rbf liftFun(cent, rbf_type);

    /* Build reference signal */
    const float Tmax = 3;
    const int Nsim = Tmax/dt;
    float ymin, ymax;
    Vector2d x0;
    MatrixXd yrr(Nsim, 1);
    int REF = 2; // Select type of reference signal (cos(2)/step(1))
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



    /* Closed loop simulation */
    Motor.state << 0.0905, 0.0653;
    MotorNom.state << 0.0905, 0.0653;
    VectorXd zeta(3, 1); zeta << 0.0653, 0.0, 0.1;
    VectorXd xlift;

    MatrixXd X(Nsim+1, n); X(0, seq(0, n-1)) = Motor.state.transpose();
    MatrixXd Xnom(Nsim+1, n); Xnom(0, seq(0, n-1)) = MotorNom.state.transpose();
    MatrixXd U(Nsim, 1);
    MatrixXd CPUtime(Nsim, 1);

    // Loop which would be running in real time on the rocket
    for (int i = 0; i < Nsim; ++i) {
        qpOASES::real_t tic = qpOASES::getCPUtime();
        // Current value of the reference signal
        double yr = yrr(i, 0);

        // Simulate closed loop feedback control
        xlift = liftFun.liftState(zeta);
        double u = MPC.getOptVal(xlift, yr);
        Motor.updateState(u);
        zeta(2, 0) = zeta(0,0); zeta(1,0) = u; zeta(0,0) = Cy*Motor.state;
        qpOASES::real_t toc = qpOASES::getCPUtime();
        MotorNom.updateState(0);

        // Store data
        X(i+1, seq(0, n-1)) = Motor.state.transpose();
        Xnom(i+1, seq(0, n-1)) = MotorNom.state.transpose();
        U(i, 0) = u;
        CPUtime(i, 0) = toc-tic;
        
        if ((i+1)%10 == 0) {
            cout << "Closed-Loop simulation: iteration " << i+1 << " out of " << Nsim << endl;
        }
    }

    // Store data in csv files
    const string referenceFile = "C:/Users/timme/OneDrive/Bureaublad/Tu Delft/DARE/Control Algorithm/Control Algorithm Tool/ControlSoftware/datafiles/reference.csv";
    const string stateFile = "C:/Users/timme/OneDrive/Bureaublad/Tu Delft/DARE/Control Algorithm/Control Algorithm Tool/ControlSoftware/datafiles/state.csv";
    const string controlFile = "C:/Users/timme/OneDrive/Bureaublad/Tu Delft/DARE/Control Algorithm/Control Algorithm Tool/ControlSoftware/datafiles/control.csv";
    const string nomStateFile = "C:/Users/timme/OneDrive/Bureaublad/Tu Delft/DARE/Control Algorithm/Control Algorithm Tool/ControlSoftware/datafiles/nomState.csv";
    const string cpuTimeFile = "C:/Users/timme/OneDrive/Bureaublad/Tu Delft/DARE/Control Algorithm/Control Algorithm Tool/ControlSoftware/datafiles/cpuTime.csv";

    saveToFile(yrr, yrr.rows(), yrr.cols(), referenceFile);
    saveToFile(X, X.rows(), X.cols(), stateFile);
    saveToFile(U, U.rows(), U.cols(), controlFile);
    saveToFile(Xnom, Xnom.rows(), Xnom.cols(), nomStateFile);
    saveToFile(CPUtime, CPUtime.rows(), CPUtime.cols(), cpuTimeFile);

    return 0;
}
