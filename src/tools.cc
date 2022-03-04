///
/// Gecko - Tools
/// Author: Olivier Stasse
/// Copyright: LAAS-CNRS
/// Date: 2022
///
///
#include <iostream>
#include <qp_formulations/tools.hh>

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
      if (aT.dimension(2)>1)
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
    if (i0!=aT.dimension(0)-1) std::cout << std::endl;
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
void extend_matrices(Eigen::MatrixXd &S,
                     Eigen::Tensor<double, 3> &U,
                     unsigned int N,
                     Eigen::MatrixXd &A,
                     Eigen::MatrixXd &B)
{
  Index n = B.rows();
  Index m = B.cols();

  MatrixXd s( n * N, n);
  s.setZero();
  Tensor<double, 3> u(n*N,N,m);
  std::cout << "u.shape=("<< n*N << ","<< N << ","<<m<<")"<<std::endl;
  u.setZero();

  s.block(0,0,n,n) = A;
  for(Index ind_n=0;ind_n <n;ind_n++)
    for(Index ind_m=0;ind_m <m;ind_m++)
      u(ind_n,0,ind_m) = B(ind_n,ind_m);

  display(u);
  for (Index i =1; i <N; i++)
  {
    
    for (Index j=0; j<m;j++)
    {
      Eigen::array<Index,3> offsets={(n*(i-1)), 0,j};
      Eigen::array<Index,3> extents={n, i,1};
      Eigen::Tensor<double,3> sub_u= u.slice(offsets,extents);
      Eigen::MatrixXd sub_u_m=Tensor_to_Matrix(sub_u,n,i);

      std::cout << "i: " << i
                << " j: " << j << "(" << n << "," << i << ")"
                << " sub_u_m:" <<sub_u_m << std::endl;

      Eigen::MatrixXd sub_u_adot = A*sub_u_m;
      std::cout << " A:" << A << std::endl
                << " sub_a_dot:" << sub_u_adot << std::endl;
    }
    //    u(n * i;n * (ind_N+1); 
  }
  //  u.block(0,0,n,m) = Matrix_to_Tensor(B,n,1,1,m);
  std::cout << "u: "<< u << std::endl;
}

} // end of tools namespace
} // end of gecko namespace
