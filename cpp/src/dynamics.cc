///
/// Gecko - Tools
/// Author: Olivier Stasse
/// Copyright: LAAS-CNRS
/// Date: 2022
///
///
#include <iostream>
#include <mpc-interface/dynamics.hh>

namespace gecko {
namespace tools {

using namespace Eigen;

ControlSystem::ControlSystem(
    std::vector<std::string> &input_names,
    std::vector<std::string> &state_names, Eigen::MatrixXd &A,
    Eigen::MatrixXd &B, std::vector<std::string> &axes,
    std::function<void(std::shared_ptr<ExtendedSystem> shr_ext_sys,
                       std::map<std::string, int> &kargs)>
        how_to_update_matrices,
    bool time_variant) {
  input_names_ = input_names;
  state_names_ = state_names;
  A_ = A;
  B_ = B;
  axes_ = axes;
  time_variant_ = time_variant;

  how_to_update_matrices_ = how_to_update_matrices;
}

void ControlSystem::update_matrices(std::shared_ptr<ExtendedSystem> shr_ext_sys,
                                    std::map<std::string, int> &kargs) {
  how_to_update_matrices_(shr_ext_sys, kargs);
}

ExtendedSystem::ExtendedSystem(
    std::vector<std::string> &input_names,
    std::vector<std::string> &state_names, std::string &state_vector_name,
    Tensor<double, 3> S, Tensor<double, 4> U, std::vector<std::string> &axes,
    std::function<void(Eigen::Tensor<double, 3> &A, Eigen::Tensor<double, 4> &B,
                       unsigned int, Eigen::MatrixXd &, Eigen::MatrixXd &)>
        how_to_update_ext_matrices,
    bool time_variant)
    : matrices_(U_, S_) {
  input_names_ = input_names;
  state_names_ = state_names;
  state_vector_name_ = state_vector_name;
  axes_ = axes;
  S_ = S;
  U_ = U;

  time_variant_ = time_variant;

  identify_domain(input_names, state_names);

  how_to_update_ext_matrices_ = how_to_update_ext_matrices;
}

// TODO
// void ExtendedSystem::
// from_control_system(cls,
//                     CcontrolSystem,
//                     state_vector_name,
//                     horizon_length)
// {
//   tools::extend_matrices(S,U,horizon_length,
//                          control_system.A,
//                          control_system,B);
//   if (control_system.

// }

void ExtendedSystem::identify_domain(std::vector<std::string> &input_name,
                                     std::vector<std::string> &state_names) {
  /// Build domain ID
  std::map<std::string, int> ldomain_ID;
  for (std::size_t i = 0; i < input_name.size(); i++)
    ldomain_ID[input_name[i]] = static_cast<int>(i);

  std::string state_vec_name_dom(state_vector_name_ + "0");
  ldomain_ID[state_vec_name_dom] = static_cast<int>(input_name.size());

  /// Build state ID
  std::map<std::string, int> lstate_ID;
  for (std::size_t i = 0; i < state_names.size(); i++)
    lstate_ID[state_names[i]] = static_cast<int>(i);

  if (axes_.size() == 0) {
    domain_ID_ = ldomain_ID;
    state_ID_ = lstate_ID;
  } else {
    for (std::size_t axes_ind = 0; axes_ind < axes_.size(); axes_ind++) {
      for (auto domain_ID_it = ldomain_ID.begin();
           domain_ID_it != ldomain_ID.end(); domain_ID_it++)
        domain_ID_[domain_ID_it->first + axis_[axes_ind]] =
            domain_ID_it->second;

      for (auto state_ID_it = lstate_ID.begin(); state_ID_it != lstate_ID.end();
           state_ID_it++)
        state_ID_[state_ID_it->first + axis_[axes_ind]] = state_ID_it->second;
    }
  }
}

void ExtendedSystem::set_sizes() {
  // Merge the two maps inside the all_variables_one.
  all_variables_.insert(domain_ID_.begin(), domain_ID_.end());
  all_variables_.insert(state_ID_.begin(), state_ID_.end());
}

// TODO
void ExtendedSystem::update_sizes() {
  if (time_variant_) {
  }
}

}  // namespace tools
}  // namespace gecko
