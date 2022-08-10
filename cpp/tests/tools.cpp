#include <iostream>

// Boost Headers
#include <boost/test/unit_test.hpp>

// Project headers
#include <mpc-interface/tools.hh>

using namespace Eigen;
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_extend_matrices_body)
{

  Eigen::MatrixXd A(3,3);
  A.row(0) << 1. , 0.1 ,0.005;
  A.row(1) << 0. ,   1. ,0.1;
  A.row(2) << 0.  ,  0. ,   1.;
  std::cout << A << std::endl;

  Eigen::MatrixXd B(3,1);
  B.row(0) << 0.00016667;
  B.row(1) << 0.005;
  B.row(2) << 0.1;
  std::cout << B << std::endl;

  Eigen::Tensor<double, 3> S;
  Eigen::Tensor<double, 4> U;
  unsigned int N=9;

  gecko::tools::extend_matrices(S,U,N,A,B);
}

BOOST_AUTO_TEST_CASE(test_extend_matrices_dynamics)
{
  Eigen::MatrixXd A=MatrixXd::Zero(8,8);
  A.block(0,1,7,7)=MatrixXd::Identity(7,7);
  A.block(7,0,1,8)=MatrixXd::Ones(1,8);
  Eigen::MatrixXd B=MatrixXd::Ones(8,6);

  Eigen::Tensor<double, 3> S;
  Eigen::Tensor<double, 4> U;
  unsigned int N=20;

  gecko::tools::extend_matrices(S,U,N,A,B);

}


BOOST_AUTO_TEST_SUITE_END()
