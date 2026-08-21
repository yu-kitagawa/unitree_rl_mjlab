#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <eigen3/Eigen/Dense>
#ifdef G1_NAVIGATION_WITH_ROS2
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#endif

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"

class State_Navigation : public FSMState
{
public:
    State_Navigation(int state_mode, std::string state_string);
    ~State_Navigation();

    void enter() override;
    bool prepare_enter() override;
    void cancel_prepare_enter() override;
    void run() override;
    void exit() override;

    std::vector<float> command_observation() const;
    std::vector<float> marker_observation() const;
    std::vector<float> future_path_observation() const;

    static State_Navigation* instance();

private:
    struct LocalizationPose
    {
        Eigen::Vector3f position{Eigen::Vector3f::Zero()};
        Eigen::Quaternionf orientation{Eigen::Quaternionf::Identity()};
        float heading{0.0f};
    };

    struct TrajectoryPose
    {
        Eigen::Vector2f position{Eigen::Vector2f::Zero()};
        float yaw{0.0f};
    };

#ifdef G1_NAVIGATION_WITH_ROS2
    void odometry_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
#endif
    bool read_latest_glim_odometry(
        LocalizationPose& pose,
        float& age_seconds) const;
    bool read_simulator_pose(LocalizationPose& pose);
    bool read_localization_pose(
        LocalizationPose& pose,
        std::string& source,
        float& age_seconds);
    void reset_navigation_state();
    void generate_trajectory();
    std::size_t trajectory_index(float offset_seconds = 0.0f) const;
    float trajectory_elapsed_seconds() const;
    void update_navigation_state();
    void update_marker_observation(float camera_heading);
    void update_future_path_observation(float robot_heading);
    void set_safe_stand_observations();
    void open_pose_log();
    void close_pose_log();
    void write_target_path_log() const;
    void write_pose_log(
        bool localization_available,
        const std::string& source,
        const LocalizationPose& localization_pose,
        float robot_heading,
        const Eigen::Vector2f& reference_position_body,
        float reference_heading_error,
        float position_error);

    float root_heading() const;
    float camera_heading() const;
    static float wrap_to_pi(float angle);

    static State_Navigation* instance_;

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::thread policy_thread;
    std::atomic_bool policy_thread_running{false};
    std::atomic_bool action_ready{false};

#ifdef G1_NAVIGATION_WITH_ROS2
    std::shared_ptr<rclcpp::Node> ros_node;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription;
    std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> ros_executor;
    std::thread ros_thread;

    mutable std::mutex odometry_mutex;
    Eigen::Vector3f latest_odometry_position{Eigen::Vector3f::Zero()};
    Eigen::Quaternionf latest_odometry_orientation{Eigen::Quaternionf::Identity()};
    std::chrono::steady_clock::time_point latest_odometry_received_at{};
    bool has_odometry{false};
#endif
    unitree::robot::go2::subscription::SportModeState::SharedPtr simulator_state;

    mutable std::mutex joint_command_mutex;
    std::vector<float> latest_policy_action;
    std::vector<float> latest_processed_action_joints;
    std::vector<float> last_sent_action_joints;
    std::vector<int> arm_command_joint_ids;

    Eigen::Vector2f estimated_position{Eigen::Vector2f::Zero()};
    Eigen::Vector2f initial_odometry_position{Eigen::Vector2f::Zero()};
    float initial_odometry_heading{0.0f};
    bool trajectory_initialized{false};
    bool localization_warning_reported{false};
    bool goal_reached{false};
    std::array<float, 3> command{0.0f, 0.0f, 0.0f};
    std::array<float, 6> marker_pose_camera{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f};
    std::array<float, 8> future_path_poses{0.0f, 0.0f, 1.0f, 0.0f,
                                           0.0f, 0.0f, 1.0f, 0.0f};

    std::vector<TrajectoryPose> trajectory_poses;
    std::vector<float> trajectory_linear_velocities;
    std::vector<float> trajectory_angular_velocities;
    std::chrono::steady_clock::time_point trajectory_started_at{};
    std::mt19937 trajectory_rng;

    std::ofstream pose_log;
    std::chrono::steady_clock::time_point navigation_started_at{};
    std::filesystem::path pose_log_path{"log/navigation_pose.csv"};
    std::filesystem::path target_path_log_path{"log/navigation_target_path.csv"};
    std::size_t pose_log_sample_count{0};
    std::size_t pose_log_flush_interval{5};
    bool pose_log_enabled{true};

    std::string localization_source{"auto"};
    std::string active_localization_source;
    std::string odometry_topic{"/glim_ros/odom"};
    std::string simulator_state_topic{"rt/sportmodestate"};
    float odometry_timeout{0.5f};
    TrajectoryPose goal;

    std::array<float, 2> reference_times{1.0f, 2.0f};
    float motion_duration{3.0f};
    float stop_hold_duration{1.0f};
    float start_ramp_duration{0.8f};
    float stop_ramp_duration{0.8f};
    std::array<int, 2> num_segments_range{1, 3};
    float min_segment_duration{0.7f};
    float min_radius{0.15f};
    float straight_probability{0.05f};
    float curvature_exponent{1.0f};
    std::array<float, 2> se2_speed_range{0.20f, 0.70f};
    float characteristic_length{0.45f};
    float tracking_gain{3.0f};
    float max_linear_speed{1.0f};
    float max_angular_speed{1.6f};
    float standing_probability{0.0f};
    float trajectory_step_dt{0.02f};
    unsigned int trajectory_seed{0U};

    // Retained only to supply the old marker observation during migration.
    float stand_off_distance{0.7f};
    float camera_forward_offset{0.05f};
    float marker_camera_height_offset{-0.054f};
    float position_tolerance{0.08f};
    float heading_tolerance{0.20f};
};

REGISTER_FSM(State_Navigation)
