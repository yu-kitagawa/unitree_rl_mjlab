#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
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

    void process_goal_input();

    std::vector<float> command_observation() const;
    std::vector<float> marker_observation() const;

    static State_Navigation* instance();

private:
    struct LocalizationPose
    {
        Eigen::Vector3f position{Eigen::Vector3f::Zero()};
        Eigen::Quaternionf orientation{Eigen::Quaternionf::Identity()};
        float heading{0.0f};
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
    void update_navigation_state();
    void update_marker_observation(float camera_heading);
    void set_safe_stand_observations();
    void open_pose_log();
    void close_pose_log();
    void write_pose_log(
        bool localization_available,
        const std::string& source,
        const LocalizationPose& localization_pose,
        float robot_heading,
        const Eigen::Vector2f& goal_position_body,
        float goal_heading_error,
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
    std::vector<float> latest_command_joints;
    std::vector<float> last_sent_action_joints;

    Eigen::Vector2f estimated_position{Eigen::Vector2f::Zero()};
    Eigen::Vector2f initial_odometry_position{Eigen::Vector2f::Zero()};
    float initial_odometry_heading{0.0f};
    bool goal_initialized{false};
    bool localization_warning_reported{false};
    bool position_reached{false};
    bool goal_reached{false};
    std::array<float, 3> command{0.0f, 0.0f, 0.0f};
    std::array<float, 6> marker_pose_camera{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f};

    std::ofstream pose_log;
    std::chrono::steady_clock::time_point navigation_started_at{};
    std::filesystem::path pose_log_path{"log/navigation_pose.csv"};
    std::size_t pose_log_sample_count{0};
    std::size_t pose_log_flush_interval{5};
    bool pose_log_enabled{true};

    std::string localization_source{"auto"};
    std::string active_localization_source;
    std::string odometry_topic{"/glim_ros/odom"};
    std::string simulator_state_topic{"rt/sportmodestate"};
    float odometry_timeout{0.5f};
    struct NavigationGoal
    {
        Eigen::Vector2f position{1.0f, 0.0f};
        float yaw{0.0f};
    };
    NavigationGoal goal;
    std::atomic_bool goal_input_requested{false};
    std::atomic_bool goal_input_active{false};
    std::atomic_bool goal_input_ready{false};
    std::atomic_bool goal_input_cancelled{false};
    std::mutex goal_input_mutex;
    bool prompt_goal_on_entry{true};
    float stand_off_distance{0.7f};
    float camera_forward_offset{0.05f};
    float marker_camera_height_offset{-0.054f};
    float position_control_stiffness{1.0f};
    float heading_control_stiffness{1.0f};
    float min_approach_speed{0.35f};
    float position_tolerance{0.08f};
    float heading_tolerance{0.20f};
    std::array<float, 2> lin_vel_x_range{-0.5f, 1.0f};
    std::array<float, 2> lin_vel_y_range{-0.5f, 0.5f};
    std::array<float, 2> ang_vel_z_range{-1.0f, 1.0f};
};

REGISTER_FSM(State_Navigation)
