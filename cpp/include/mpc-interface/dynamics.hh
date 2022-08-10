///
/// Gecko - Tools
/// Author: Olivier Stasse
/// Copyright: LAAS-CNRS
/// Date: 2022
///
///

#ifndef GECKO_DYNAMICS_H_
#define GECKO_DYNAMICS_H_

#include <Eigen/Eigen>
#include <functional>
#include <memory>

#if !EIGEN_HAS_CXX11
#define XSTR(x) STR(x)
#define STR(x) #x
#pragma message "The value of EIGEN_MAX_CPP_VER is " XSTR(EIGEN_MAX_CPP_VER)
#pragma message "The value of EIGEN_COMP_CXXVER is " XSTR(EIGEN_COMP_CXXVER)
#error EIGEN_HAS_CXX11 is required.
#endif

#include <unsupported/Eigen/CXX11/Tensor>

namespace gecko {
namespace tools {

/// *{
/// *}

class ExtendedSystem;

class ControlSystem {
 public:
  ControlSystem(std::vector<std::string> &input_names,
                std::vector<std::string> &state_names, Eigen::MatrixXd &A,
                Eigen::MatrixXd &B, std::vector<std::string> &axes,
                std::function<void(std::shared_ptr<ExtendedSystem> shr_ext_sys,
                                   std::map<std::string, int> &kargs)>
                    how_to_update_matrices,
                bool time_variant = false);

  void update_matrices(std::shared_ptr<ExtendedSystem> shr_ext_sys,
                       std::map<std::string, int> &kargs);

 protected:
  /// Store the names of the inputs
  std::vector<std::string> input_names_;

  /// Store the names of the states
  std::vector<std::string> state_names_;

  /// A:
  Eigen::MatrixXd A_;

  /// B
  Eigen::MatrixXd B_;

  std::vector<std::string> axes_;

  /// Is the system time variant ?
  bool time_variant_;

  /// Ref to method to call
  std::function<void(std::shared_ptr<ExtendedSystem> shr_ext_sys,
                     std::map<std::string, int> &kargs)>
      how_to_update_matrices_;
};
/// *{
/// \class ExtendenSystem
/// This class works a dynamics of the form:
/// \$ x = S*x0 + U*u \$
/// where
/// S: is a 3d tensor of shape \$ [N,n,n] \$.
/// U: is a 4d tensor of shpae \$ [m,N,N,n] \$.
/// with \$ N\$ the horizon length, \$ n\$ the number of states,
/// \$ m \$ number of inputs and \$ p_u \$ the number of ctions predicted
/// for each input.
/// *}

class ExtendedSystem {
 public:
  /// Constructor
  ExtendedSystem(
      std::vector<std::string> &input_names,
      std::vector<std::string> &state_names, std::string &state_vector_name,
      Eigen::Tensor<double, 3> S, Eigen::Tensor<double, 4> U,
      std::vector<std::string> &axis,
      std::function<void(Eigen::Tensor<double, 3> &, Eigen::Tensor<double, 4> &,
                         unsigned int, Eigen::MatrixXd &, Eigen::MatrixXd &)>
          how_to_update_ext_matrices,
      bool time_variant = true);

  void identify_domain(std::vector<std::string> &input_name,
                       std::vector<std::string> &state_names);

 protected:
  /// Store the list of axis.
  std::vector<std::string> axis_;

  /// Store the names of the inputs
  std::vector<std::string> input_names_;

  /// Store the names of the states
  std::vector<std::string> state_names_;

  /// State vector name
  std::string state_vector_name_;

  /// Matrices
  /// State related matrix
  Eigen::Tensor<double, 3> S_;

  /// Command related matrixo
  Eigen::Tensor<double, 4> U_;

  /// Domain ID
  std::map<std::string, int> domain_ID_;

  /// State ID
  std::map<std::string, int> state_ID_;

  /// All variables
  std::map<std::string, int> all_variables_;

  /// List of axis in the model
  std::vector<std::string> axes_;

  /// Is the system time variant ?
  bool time_variant_;

  /// Matrices is a tuple of tensors
  std::tuple<Eigen::Tensor<double, 4> &, Eigen::Tensor<double, 3> &> matrices_;

  std::function<void(Eigen::Tensor<double, 3> &A, Eigen::Tensor<double, 4> &B,
                     unsigned int, Eigen::MatrixXd &, Eigen::MatrixXd &)>
      how_to_update_ext_matrices_;

  /// Populate the all_variables member.
  /// TODO ? Change the name of set_sizes
  void set_sizes();

  /// TODO:
  void update_sizes();
};

}  // namespace tools
}  // namespace gecko
#endif
