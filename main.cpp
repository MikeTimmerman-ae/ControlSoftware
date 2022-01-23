#include <iostream>
#include <fstream>
#include <vector>
#include<math.h>
#include<string>

#define PI 3.14159265

const int n_states = 103;
const float dt = 0.01;

using namespace Eigen;
using namespace std;

void loadAlift(Matrix<double, n_states, n_states> &AliftRef, string FileName) {
    ifstream AliftFile;

    AliftFile.open(FileName);
    int i = 0;
    int j = 0;
    for (int k = 0; k < n_states; k++) {
        string line;
        getline(AliftFile, line, '\n');
        //cout << line << endl;
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

void loadBlift(Matrix<double, n_states, 1> &BliftRef, string FileName) {
    ifstream BliftFile;
    BliftFile.open(FileName);
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

template <typename Derived>
void bdiag(const MatrixBase<Derived> &A, const MatrixBase<Derived> &B, int N) {
    int m = B.rows();
    int n = B.cols();
    for (int i = 0; i < N; ++i) {
        const_cast< MatrixBase<Derived>& >(A)(seq(m*i, (i+1)*m -1), seq(n*i, (i+1)*n -1)) = B;
    }
}


class Controller
{
    public:
        int Np, p;
        double *Ab;
        //double *Xlb, *Xub;
        //double *Ulb, *Uub;
        //double *ulin;
        double *M1, *M2;
        double *C;
        double Q, Qp, d;

        Controller(Matrix<double, n_states, n_states> &A, Matrix<double, n_states, 1> &B, MatrixXd &C, double d, double Q, double R, double QN,
                   int Np, MatrixXd &Ulb, MatrixXd &Uub, MatrixXd &Xlb, MatrixXd &Xub, VectorXd &ulin, VectorXd &qlin, string solver = "qpoases") {

            const int n = A.rows();             // Number of states
            const int m = B.cols();             // Number of control inputs
            const int p = C.rows();             // Number of outputs

            VectorXd x0(n);               // Dummy Variable

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
            MatrixXd Ab_temp = MatrixXd::Identity((Np+1)*n, n);
            MatrixXd Bb_temp((Np+1)*n, Np*m);
            for (int i = 1; i <= Np; ++i) {
                Ab_temp(seq(i*n, (i+1)*n-1), all) = Ab_temp(seq((i-1)*n, i*n-1), all) * A;
                Bb_temp(seq(i*n, (i+1)*n-1), all) = A * Bb_temp(seq((i-1)*n, i*n-1), all);
                Bb_temp(seq(i*n, (i+1)*n-1), seq((i-1)*m, i*m-1)) = B;
            }
            MatrixXd Ab = Ab_temp(seq(n, last), all);       // Copying these into a new matrix is suboptimal
            MatrixXd Bb = Bb_temp(seq(n, last), all);       // Pls fix this lol
            // cout << Bb(seq(2*n, 3*n-1), seq(0,10));

            // Build the controller
            SparseMatrix<double> Qb(p*Np,p*Np);
            for (int i = 0; i < Np*p; ++i) {Qb.insert(i, i) = Q;}
            Qb.insert(p*Np-1, p*Np-1) = QN;

            MatrixXd Cb(Np*C.rows(), Np*C.cols()); bdiag(Cb, C, Np);

            MatrixXd Rb = MatrixXd::Identity(Np, Np)*R;

            //MatrixXd M1 = 2* ( ( Bb.transpose()* (Cb.transpose()*Qb*Cb) ) *Ab );
            //MatrixXd M2 = (-2* (Qb*Cb) *Bb).transpose();

            // Bounds on the states
            MatrixXd Aineq_temp(n*Np*2, Np);
            MatrixXd bineq_temp(n*Np*2, 1);

            Xub = Xub.reshaped(); Xlb = Xlb.reshaped();
            Uub = Uub.reshaped(); Ulb = Ulb.reshaped();

            Aineq_temp(seq(0, Np*n-1), all) = Bb; Aineq_temp(seq(Np*n, 2*Np*n-1), all) = -Bb;
            bineq_temp(seq(0, Np*n-1), all) = Xub - Ab*x0; bineq_temp(seq(Np*n, 2*Np*n-1), all) = -Xlb + Ab*x0;

            vector<int> indices = NotNanIndex(bineq_temp, bineq_temp.rows());
            MatrixXd Aineq = Aineq_temp(indices, all);
            MatrixXd bineq = bineq_temp(indices, all);

            MatrixXd H = 2*(Bb.transpose()*Cb.transpose()*Qb*Cb*Bb + Rb);
            MatrixXd f = (2*x0.transpose()*Ab.transpose()*Cb.transpose()*Qb*Cb*Bb).transpose() + ulin + Bb.transpose()*(Cb.transpose()*qlin);
            H = (H+H.transpose())/2;

            // Initialize controller with qpOASES





        }


        double getOptInput()
        {
            return 0;
        }

};

int main()
{
    /* Importing System Dynamics Data */;
    string FileAlift = "Alift.csv";
    string FileBlift = "Blift.csv";

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
    MatrixXd C(1, n_states); C(0, 0) = 1;
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

    /*
    for(int i=0;i<n_states;++i){
		for(int j=0;j<n_states;++j){
			cout<<C[i][j]<<' ';
		}
		cout<<endl;
	}
    */
    /*
    for(int i=0;i<n_states;++i){
        cout<< xlift_min[i] <<endl;
	}
    */

    return 0;
}
