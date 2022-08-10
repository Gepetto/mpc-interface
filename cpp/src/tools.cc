///
/// Gecko - Tools
/// Author: Olivier Stasse
/// Copyright: LAAS-CNRS
/// Date: 2022
///
///

#ifndef QP_FORMULATIONS_TOOLS_H_
#define QP_FORMULATIONS_TOOLS_H_

// Standard C++ include
#include <iostream>

// This repository includes
#include <mpc-interface/tools.hh>
#include <mpc-interface/dynamics.hh>

namespace gecko {
namespace tools {

using namespace Eigen;

template<typename T>
using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic>;

template<typename Scalar,int rank, typename sizeType>
auto Tensor_to_Matrix(const Eigen::Tensor<Scalar,rank> &tensor,const sizeType rows,const sizeType cols)
{
    return Eigen::Map<const MatrixType<Scalar>> (tensor.data(), rows,cols);
}


template<typename Scalar, typename... Dims>
auto Matrix_to_Tensor(const MatrixType<Scalar> &matrix, Dims... dims)
{
    constexpr int rank = sizeof... (Dims);
    return Eigen::TensorMap<Eigen::Tensor<const Scalar, rank>>(matrix.data(), {dims...});
}

void display(Tensor<double,3> &aT)
{
  std::cout << "("
            << aT.dimension(0) <<","
            << aT.dimension(1) <<","
            << aT.dimension(2) <<")=[";
  for(Index i0=0;i0<aT.dimension(0);i0++)
  {
    std::cout <<"[";
    for(Index i1=0;i1<aT.dimension(1);i1++)
    {
      std::cout <<"[";
      for(Index i2=0;i2<aT.dimension(2);i2++)
      {
            std::cout << aT(i0,i1,i2);
            if (i2!=aT.dimension(2)-1)
              std::cout << ",";
      }
      std::cout <<"]";
      if (i1!=aT.dimension(1)-1) std::cout << std::endl;
    }
    std::cout <<"]";
    if (i0!=aT.dimension(0)-1) std::cout << std::endl << std::endl;
  }
  std::cout <<"]"<<std::endl;
}

void display(Tensor<double,4> &aT)
{
  std::cout << "("
            << aT.dimension(0) <<","
            << aT.dimension(1) <<","
            << aT.dimension(2) <<","
            << aT.dimension(3) <<")=[";
  for(Index i0=0;i0<aT.dimension(0);i0++)
  {

      std::cout <<"[";
      for(Index i1=0;i1<aT.dimension(1);i1++)
      {
        std::cout <<"[";
        for(Index i2=0;i2<aT.dimension(2);i2++)
        {
          if (aT.dimension(3)>1)
            std::cout <<"[";
          for(Index i3=0;i3<aT.dimension(3);i3++)
          {
            std::cout << aT(i0,i1,i2,i3);
            if (i3!=aT.dimension(3)-1)
              std::cout << ",";
          }
          if (aT.dimension(3)>1)
            std::cout <<"]";
          if (i2!=aT.dimension(2)-1) std::cout << std::endl;
        }
        std::cout <<"]";
        if (i1!=aT.dimension(1)-1) std::cout << std::endl;
      }
      std::cout <<"]"<<std::endl;
  }
  std::cout <<"]"<<std::endl;
}
void extend_matrices(Eigen::Tensor<double, 3> &S,
                     Eigen::Tensor<double, 4> &U,
                     unsigned int N,
                     Eigen::MatrixXd &A,
                     Eigen::MatrixXd &B)
{
  Index n = B.rows();
  Index m = B.cols();

  S.resize(N,n,n);
  MatrixXd s( n * N, n);
  s.setZero();
  Tensor<double, 3> u(n*N,N,m);
  u.setZero();

  s.block(0,0,n,n) = A;
  for(Index ind_n=0;ind_n <n;ind_n++)
    for(Index ind_m=0;ind_m <m;ind_m++)
      u(ind_n,0,ind_m) = B(ind_n,ind_m);

  for (Index i =1; i <N; i++)
  {

    for (Index j=0; j<m;j++)
    {
      Eigen::array<Index,3> offsets={(n*(i-1)), 0,j};
      Eigen::array<Index,3> extents={n, i,1};
      Eigen::Tensor<double,3> sub_u= u.slice(offsets,extents);
      Eigen::MatrixXd sub_u_m=Tensor_to_Matrix(sub_u,n,i);

      Eigen::MatrixXd sub_u_adot = A*sub_u_m;
      Eigen::array<Index,3> offsets_concat={n*i, 0,j};
      Eigen::array<Index,3> extents_concat={n, i,1};

      u.slice(offsets_concat,extents_concat) =
          Matrix_to_Tensor(sub_u_adot,n,i,1);
    }

    Eigen::array<Index,3> offsets_B={n*i, i,0};
    Eigen::array<Index,3> extents_B={n, 1,m};

    u.slice(offsets_B,extents_B) =
        Matrix_to_Tensor(B,n,1,m);


    Eigen::array<Index,3> offsets_disp={n*i, 0,0};
    Eigen::array<Index,3> extents_disp={n, i,m};

    Eigen::Tensor<double,3> u_disp=u.slice(offsets_disp,extents_disp);

    s.block(n*i,0,n,n) = A*s.block(n*(i-1),0,n,n);

  }

  //  S = np.dstack([s[i : n * N : n, :] for i in range(n)])
  for (Index i=0;i<N;i++)
  {
    Eigen::array<Index,3> offsets_S={i, 0,0};
    Eigen::array<Index,3> extents_S={1, n,n};

    MatrixXd sb = s.block(n*i,0,n,n);
    MatrixXd sb_t=sb.transpose();
    S.slice(offsets_S,extents_S)=Matrix_to_Tensor(sb_t,1,n,n);
  }

  //    U = [np.dstack([u[i : n * N : n, :, j] for i in range(n)]) for j in range(m)]
  U.resize(m,N,N,n);
  for(Index j=0;j<m;j++)
    for(Index i=0,lU_i=0;i<n*N;i+=n,lU_i++)
      for(Index k=0;k<N;k++)
        for(Index l=0;l<n;l++)
          U(j,lU_i,k,l)=u(i+l,k,j);

}

void update_step_matrices
(std::shared_ptr<ExtendedSystem> /*shr_ext_sys*/,
 std::map<std::string,double> & /*kargs*/)
{

}

void get_system_matrices(std::string & /*system*/)
{

}

} // end of tools namespace
} // end of gecko namespace
#endif
