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
  Tensor<double, 4> u(n*N,N,N,m);
  u.setZero();

  s.block(0,0,n,n) = A;
  //  u.block(0,0,n,m) = Matrix_to_Tensor(B,n,1,1,m);
  std::cout << "u: "<< u << std::endl;
}

} // end of tools namespace
} // end of gecko namespace
