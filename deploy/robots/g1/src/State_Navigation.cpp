#include "State_Navigation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "unitree_articulation.h"

State_Navigation* State_Navigation::instance_ = nullptr;

namespace
{

std::array<float, 2> read_range(
    const YAML::Node& node,
    const std::array<float, 2>& fallback)
{
    if (!node || !node.IsSequence() || node.size() != 2) {
        return fallback;
    }
    return {node[0].as<float>(), node[1].as<float>()};
}

float quaternion_heading(const Eigen::Quaternionf& quat)
{
    const Eigen::Vector3f forward = quat * Eigen::Vector3f::UnitX();
    return std::atan2(forward.y(), forward.x());
}

}  // namespace

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(navigation_commands)
{
    (void)env;
    (void)params;
    return State_Navigation::instance()->command_observation();
}

REGISTER_OBSERVATION(navigation_marker_pose_camera)
{
    (void)env;
    (void)params;
    return State_Navigation::instance()->marker_observation();
}

REGISTER_OBSERVATION(navigation_phase)
{
    const auto command = State_Navigation::instance()->command_observation();
    const float command_norm = std::sqrt(
        command[0] * command[0] +
        command[1] * command[1] +
        command[2] * command[2]
    );
    if (command_norm < 0.1f) {
        return std::vector<float>{0.0f, 0.0f};
    }

    const float period = params["period"].as<float>();
    const float phase = std::fmod(env->episode_length * env->step_dt, period) / period;
    return std::vector<float>{
        std::sin(phase * 2.0f * static_cast<float>(M_PI)),
        std::cos(phase * 2.0f * static_cast<float>(M_PI)),
    };
}

REGISTER_OBSERVATION(navigation_arm_pose_commands)
{
    const std::string command_name = params["command_name"].as<std::string>();
    const std::string target_name = command_name + "_target";
    const auto command_cfg = env->cfg["commands"][command_name];

    auto command = env->get_command(command_name);
    auto target = env->get_command(target_name);
    if (command.empty()) {
        const std::string default_pose =
            command_cfg["default_pose"].as<std::string>("down");
        command = command_cfg["poses"][default_pose].as<std::vector<float>>();
        target = command;
        env->set_command(target_name, target);
    }
    if (target.empty()) {
        target = command;
        env->set_command(target_name, target);
    }
    if (command.size() != target.size()) {
        throw std::runtime_error(
            "Arm pose command and target must have the same dimension."
        );
    }

    const float max_joint_speed =
        command_cfg["max_joint_speed"].as<float>(0.15f);
    if (max_joint_speed <= 0.0f) {
        throw std::runtime_error("Arm pose max_joint_speed must be positive.");
    }
    const float max_step = max_joint_speed * env->step_dt;
    for (size_t i = 0; i < command.size(); ++i) {
        const float error = target[i] - command[i];
        command[i] += std::clamp(error, -max_step, max_step);
    }

    env->set_command(command_name, command);
    return command;
}

}  // namespace mdp
}  // namespace isaaclab

State_Navigation::State_Navigation(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    if (instance_ != nullptr) {
        throw std::runtime_error("Only one Navigation state can be configured.");
    }
    instance_ = this;

    const auto state_cfg = param::config["FSM"][state_string];
    const auto policy_dir = param::parser_policy_dir(
        state_cfg["policy_dir"].as<std::string>()
    );
    const auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    const auto nav_cfg = deploy_cfg["commands"]["navigation"];

    localization_source = nav_cfg["localization_source"].as<std::string>(
        localization_source
    );
    odometry_topic = nav_cfg["odometry_topic"].as<std::string>(odometry_topic);
    simulator_state_topic = nav_cfg["simulator_state_topic"].as<std::string>(
        simulator_state_topic
    );
    pose_log_enabled = nav_cfg["pose_log_enabled"].as<bool>(pose_log_enabled);
    pose_log_path = nav_cfg["pose_log_path"].as<std::string>(
        pose_log_path.string()
    );
    pose_log_flush_interval = nav_cfg["pose_log_flush_interval"].as<std::size_t>(
        pose_log_flush_interval
    );
    odometry_timeout = nav_cfg["odometry_timeout"].as<float>(odometry_timeout);
    goal.position.x() = nav_cfg["goal_x"].as<float>(goal.position.x());
    goal.position.y() = nav_cfg["goal_y"].as<float>(goal.position.y());
    goal.yaw = nav_cfg["goal_yaw"].as<float>(goal.yaw);
    prompt_goal_on_entry = nav_cfg["prompt_goal_on_entry"].as<bool>(
        prompt_goal_on_entry
    );
    stand_off_distance = nav_cfg["stand_off_distance"].as<float>(stand_off_distance);
    camera_forward_offset = nav_cfg["camera_forward_offset"].as<float>(camera_forward_offset);
    marker_camera_height_offset = nav_cfg["marker_camera_height_offset"].as<float>(marker_camera_height_offset);
    position_control_stiffness = nav_cfg["position_control_stiffness"].as<float>(position_control_stiffness);
    heading_control_stiffness = nav_cfg["heading_control_stiffness"].as<float>(heading_control_stiffness);
    min_approach_speed = nav_cfg["min_approach_speed"].as<float>(min_approach_speed);
    position_tolerance = nav_cfg["position_tolerance"].as<float>(position_tolerance);
    heading_tolerance = nav_cfg["heading_tolerance"].as<float>(heading_tolerance);
    lin_vel_x_range = read_range(nav_cfg["ranges"]["lin_vel_x"], lin_vel_x_range);
    lin_vel_y_range = read_range(nav_cfg["ranges"]["lin_vel_y"], lin_vel_y_range);
    ang_vel_z_range = read_range(nav_cfg["ranges"]["ang_vel_z"], ang_vel_z_range);

    if (localization_source != "auto" &&
        localization_source != "glim" &&
        localization_source != "simulator") {
        throw std::runtime_error(
            "Navigation localization_source must be auto, glim, or simulator."
        );
    }
#ifdef G1_NAVIGATION_WITH_ROS2
    const bool use_glim = localization_source != "simulator";
#else
    const bool use_glim = false;
    if (localization_source == "glim") {
        throw std::runtime_error(
            "This g1_ctrl was built without ROS 2; use simulator or auto "
            "localization, or rebuild with G1_NAVIGATION_WITH_ROS2=ON."
        );
    }
#endif
    const bool use_simulator = localization_source != "glim";
    if ((use_glim && odometry_topic.empty()) ||
        (use_simulator && simulator_state_topic.empty()) ||
        odometry_timeout <= 0.0f) {
        throw std::runtime_error("Invalid Navigation localization configuration.");
    }
    if (pose_log_enabled &&
        (pose_log_path.empty() || pose_log_flush_interval == 0)) {
        throw std::runtime_error("Invalid Navigation pose log configuration.");
    }
    if (position_tolerance <= 0.0f || stand_off_distance < 0.0f) {
        throw std::runtime_error("Invalid Navigation goal or stand-off distance.");
    }

    marker_pose_camera = {
        goal.position.x() + stand_off_distance * std::cos(goal.yaw) - camera_forward_offset * std::cos(goal.yaw),
        goal.position.y() + stand_off_distance * std::sin(goal.yaw) - camera_forward_offset * std::sin(goal.yaw),
        marker_camera_height_offset,
        0.0f,
        0.0f,
        -1.0f,
    };
    marker_pose_camera[3] = std::sqrt(
        marker_pose_camera[0] * marker_pose_camera[0] +
        marker_pose_camera[2] * marker_pose_camera[2]
    );

    auto articulation = std::make_shared<
        unitree::BaseArticulation<LowState_t::SharedPtr>
    >(FSMState::lowstate);
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(deploy_cfg, articulation);
    env->alg = std::make_unique<isaaclab::OrtRunner>(
        policy_dir / "exported" / "policy.onnx"
    );

    registered_checks.emplace_back(std::make_pair(
        [&]() -> bool { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
        FSMStringMap.right.at("Passive")
    ));

#ifdef G1_NAVIGATION_WITH_ROS2
    if (use_glim) {
        ros_node = std::make_shared<rclcpp::Node>("g1_navigation_localization");
        odometry_subscription = ros_node->create_subscription<nav_msgs::msg::Odometry>(
            odometry_topic,
            rclcpp::SensorDataQoS(),
            std::bind(&State_Navigation::odometry_callback, this, std::placeholders::_1)
        );
        ros_executor = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
        ros_executor->add_node(ros_node);
        ros_thread = std::thread([this] { ros_executor->spin(); });
    }
#endif
    if (use_simulator) {
        simulator_state = std::make_shared<
            unitree::robot::go2::subscription::SportModeState
        >(simulator_state_topic);
        simulator_state->set_timeout_ms(std::max<uint32_t>(
            1U,
            static_cast<uint32_t>(odometry_timeout * 1000.0f)
        ));
    }
    spdlog::info(
        "Navigation localization source: {} (GLIM: {}, simulator: {}, "
        "timeout: {:.2f} s).",
        localization_source,
        odometry_topic,
        simulator_state_topic,
        odometry_timeout
    );
}

State_Navigation::~State_Navigation()
{
    policy_thread_running = false;
    if (policy_thread.joinable()) {
        policy_thread.join();
    }
    close_pose_log();
#ifdef G1_NAVIGATION_WITH_ROS2
    if (ros_executor) {
        ros_executor->cancel();
    }
    if (ros_thread.joinable()) {
        ros_thread.join();
    }
    if (ros_executor && ros_node) {
        ros_executor->remove_node(ros_node);
    }
#endif
    if (instance_ == this) {
        instance_ = nullptr;
    }
}

void State_Navigation::enter()
{
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i) {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0.0f;
        lowcmd->msg_.motor_cmd()[i].tau() = 0.0f;
    }

    env->robot->update();
    reset_navigation_state();
    open_pose_log();
    env->reset();

    action_ready = false;
    policy_thread_running = true;
    policy_thread = std::thread([this] {
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desired_duration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desired_duration);
        auto sleep_until = clock::now() + dt;

        while (policy_thread_running) {
            env->robot->update();
            update_navigation_state();
            env->step();
            {
                std::lock_guard<std::mutex> lock(joint_command_mutex);
                latest_policy_action = env->action_manager->action();
                latest_command_joints = env->action_manager->processed_actions();
            }
            action_ready = true;

            std::this_thread::sleep_until(sleep_until);
            sleep_until += dt;
        }
    });
}

bool State_Navigation::prepare_enter()
{
    if (!prompt_goal_on_entry) {
        return true;
    }
    if (goal_input_ready.exchange(false)) {
        goal_input_requested = false;
        return true;
    }
    {
        std::lock_guard<std::mutex> lock(goal_input_mutex);
        if (goal_input_active || goal_input_ready) {
            return false;
        }
        goal_input_cancelled = false;
        goal_input_requested = true;
    }
    return false;
}

void State_Navigation::cancel_prepare_enter()
{
    std::lock_guard<std::mutex> lock(goal_input_mutex);
    goal_input_requested = false;
    goal_input_ready = false;
    goal_input_cancelled = true;
}

void State_Navigation::process_goal_input()
{
    if (!prompt_goal_on_entry || !goal_input_requested) {
        return;
    }

    bool expected_inactive = false;
    if (!goal_input_active.compare_exchange_strong(expected_inactive, true)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(goal_input_mutex);
        goal_input_requested = false;
    }

    NavigationGoal selected_goal = goal;
    while (true) {
        std::ostringstream prompt;
        prompt
            << "Enter Navigation goal x y yaw_rad in the robot frame at entry "
            << "(current: " << std::fixed << std::setprecision(3)
            << goal.position.x() << ' ' << goal.position.y() << ' '
            << goal.yaw << ", empty = keep current):";
        const std::string input = keyboard
            ? keyboard->getString(prompt.str())
            : std::string{};

        if (input.find_first_not_of(" \t\r\n") == std::string::npos) {
            break;
        }

        std::string normalized_input = input;
        std::replace(normalized_input.begin(), normalized_input.end(), ',', ' ');
        std::istringstream parser(normalized_input);
        float goal_x = 0.0f;
        float goal_y = 0.0f;
        float goal_yaw = 0.0f;
        std::string trailing_value;
        if ((parser >> goal_x >> goal_y >> goal_yaw) &&
            !(parser >> trailing_value) &&
            std::isfinite(goal_x) &&
            std::isfinite(goal_y) &&
            std::isfinite(goal_yaw)) {
            selected_goal.position = Eigen::Vector2f(goal_x, goal_y);
            selected_goal.yaw = wrap_to_pi(goal_yaw);
            break;
        }

        spdlog::warn(
            "Invalid Navigation goal. Enter three finite values: x y yaw_rad."
        );
    }

    {
        std::lock_guard<std::mutex> lock(goal_input_mutex);
        goal_input_requested = false;
        if (goal_input_cancelled) {
            goal_input_cancelled = false;
            goal_input_active = false;
            spdlog::warn(
                "Navigation goal input discarded because the pending "
                "transition was cancelled."
            );
            return;
        }
        goal = selected_goal;
        spdlog::info(
            "Navigation goal selected: x={:.3f}, y={:.3f}, yaw={:.3f} rad.",
            goal.position.x(),
            goal.position.y(),
            goal.yaw
        );
        goal_input_ready = true;
        goal_input_active = false;
    }
}

void State_Navigation::run()
{
    const auto arm_pose_cfg = env->cfg["commands"]["arm_pose"];
    if (arm_pose_cfg && arm_pose_cfg["poses"]) {
        const auto& joystick = FSMState::lowstate->joystick;
        std::string requested_pose;

        if (joystick.LB.pressed && joystick.up.on_pressed) {
            requested_pose = "up";
        } else if (joystick.LB.pressed && joystick.down.on_pressed) {
            requested_pose = "down";
        }

        if (!requested_pose.empty()) {
            const auto target = arm_pose_cfg["poses"][requested_pose]
                .as<std::vector<float>>();
            env->set_command("arm_pose_target", target);
            spdlog::info(
                "Navigation arm pose target: {} "
                "(max joint speed: {:.2f} rad/s)",
                requested_pose,
                arm_pose_cfg["max_joint_speed"].as<float>(0.15f)
            );
        }
    }

    if (!action_ready) {
        return;
    }
    std::lock_guard<std::mutex> lock(joint_command_mutex);
    const auto& action = latest_command_joints;
    if (action.size() != env->robot->data.joint_ids_map.size()) {
        return;
    }
    if (last_sent_action_joints.size() != env->robot->data.joint_ids_map.size()) {
        last_sent_action_joints.resize(env->robot->data.joint_ids_map.size());
    }
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i) {
        const int sdk_joint_id = env->robot->data.joint_ids_map[i];
        lowcmd->msg_.motor_cmd()[sdk_joint_id].q() = action[i];
        last_sent_action_joints[i] = lowcmd->msg_.motor_cmd()[sdk_joint_id].q();
    }
}

void State_Navigation::exit()
{
    policy_thread_running = false;
    if (policy_thread.joinable()) {
        policy_thread.join();
    }
    close_pose_log();
    action_ready = false;
}

std::vector<float> State_Navigation::command_observation() const
{
    return std::vector<float>(command.begin(), command.end());
}

std::vector<float> State_Navigation::marker_observation() const
{
    return std::vector<float>(marker_pose_camera.begin(), marker_pose_camera.end());
}

State_Navigation* State_Navigation::instance()
{
    if (instance_ == nullptr) {
        throw std::runtime_error("Navigation state has not been initialized.");
    }
    return instance_;
}

#ifdef G1_NAVIGATION_WITH_ROS2
void State_Navigation::odometry_callback(
    const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
    const auto& position = msg->pose.pose.position;
    const auto& orientation = msg->pose.pose.orientation;
    const bool pose_is_finite =
        std::isfinite(position.x) &&
        std::isfinite(position.y) &&
        std::isfinite(position.z) &&
        std::isfinite(orientation.x) &&
        std::isfinite(orientation.y) &&
        std::isfinite(orientation.z) &&
        std::isfinite(orientation.w);
    if (!pose_is_finite) {
        spdlog::warn("Ignoring a non-finite odometry pose from {}.", odometry_topic);
        return;
    }

    Eigen::Quaternionf quaternion(
        static_cast<float>(orientation.w),
        static_cast<float>(orientation.x),
        static_cast<float>(orientation.y),
        static_cast<float>(orientation.z)
    );
    if (quaternion.squaredNorm() < 1.0e-8f) {
        spdlog::warn("Ignoring an invalid odometry quaternion from {}.", odometry_topic);
        return;
    }
    quaternion.normalize();

    std::lock_guard<std::mutex> lock(odometry_mutex);
    latest_odometry_position = Eigen::Vector3f(
        static_cast<float>(position.x),
        static_cast<float>(position.y),
        static_cast<float>(position.z)
    );
    latest_odometry_orientation = quaternion;
    latest_odometry_received_at = std::chrono::steady_clock::now();
    has_odometry = true;
}
#endif

bool State_Navigation::read_latest_glim_odometry(
    LocalizationPose& pose,
    float& age_seconds) const
{
#ifdef G1_NAVIGATION_WITH_ROS2
    std::lock_guard<std::mutex> lock(odometry_mutex);
    if (!has_odometry) {
        return false;
    }

    pose.position = latest_odometry_position;
    pose.orientation = latest_odometry_orientation;
    pose.heading = quaternion_heading(pose.orientation);
    age_seconds = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - latest_odometry_received_at
    ).count();
    return true;
#else
    (void)pose;
    (void)age_seconds;
    return false;
#endif
}

bool State_Navigation::read_simulator_pose(LocalizationPose& pose)
{
    if (!simulator_state || simulator_state->isTimeout()) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(simulator_state->mutex_);
        const auto& simulator_position = simulator_state->msg_.position();
        pose.position = Eigen::Vector3f(
            simulator_position[0],
            simulator_position[1],
            simulator_position[2]
        );
    }
    pose.orientation = env->robot->data.root_quat_w;
    if (pose.orientation.squaredNorm() < 1.0e-8f) {
        return false;
    }
    pose.orientation.normalize();
    pose.heading = quaternion_heading(pose.orientation);
    return pose.position.allFinite() &&
        pose.orientation.coeffs().allFinite() &&
        std::isfinite(pose.heading);
}

bool State_Navigation::read_localization_pose(
    LocalizationPose& pose,
    std::string& source,
    float& age_seconds)
{
    const auto try_glim = [&]() {
        float glim_age = 0.0f;
        if (!read_latest_glim_odometry(pose, glim_age) ||
            glim_age > odometry_timeout) {
            return false;
        }
        source = "glim";
        age_seconds = glim_age;
        return true;
    };
    const auto try_simulator = [&]() {
        if (!read_simulator_pose(pose)) {
            return false;
        }
        source = "simulator";
        age_seconds = 0.0f;
        return true;
    };

    const std::string& selected_source = active_localization_source.empty()
        ? localization_source
        : active_localization_source;
    if (selected_source == "glim") {
        return try_glim();
    }
    if (selected_source == "simulator") {
        return try_simulator();
    }

    // Auto mode chooses once per Navigation entry. GLIM is preferred when
    // both sources are present, and the chosen coordinate frame is then held.
    return try_glim() || try_simulator();
}

void State_Navigation::reset_navigation_state()
{
    estimated_position.setZero();
    initial_odometry_position.setZero();
    initial_odometry_heading = 0.0f;
    goal_initialized = false;
    active_localization_source.clear();
    localization_warning_reported = false;
    position_reached = false;
    goal_reached = false;
    command = {0.0f, 0.0f, 0.0f};
    {
        std::lock_guard<std::mutex> lock(joint_command_mutex);
        latest_policy_action.assign(
            env->robot->data.joint_ids_map.size(),
            std::numeric_limits<float>::quiet_NaN()
        );
        latest_command_joints.assign(
            env->robot->data.joint_ids_map.size(),
            std::numeric_limits<float>::quiet_NaN()
        );
        last_sent_action_joints.assign(
            env->robot->data.joint_ids_map.size(),
            std::numeric_limits<float>::quiet_NaN()
        );
    }
    set_safe_stand_observations();
}

void State_Navigation::update_navigation_state()
{
    LocalizationPose localization_pose;
    float odometry_age = 0.0f;
    std::string pose_source;
    const bool localization_available = read_localization_pose(
        localization_pose,
        pose_source,
        odometry_age
    );
    if (!localization_available) {
        command = {0.0f, 0.0f, 0.0f};
        set_safe_stand_observations();
        const std::string waiting_for = active_localization_source.empty()
            ? localization_source
            : active_localization_source;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        write_pose_log(
            false,
            waiting_for,
            localization_pose,
            nan,
            Eigen::Vector2f(nan, nan),
            nan,
            nan
        );
        if (!localization_warning_reported) {
            spdlog::warn(
                "Navigation is waiting for fresh {} localization.",
                waiting_for
            );
            localization_warning_reported = true;
        }
        return;
    }

    if (localization_warning_reported) {
        spdlog::info(
            "Navigation localization is available from {}.",
            pose_source
        );
        localization_warning_reported = false;
    }

    if (!goal_initialized) {
        active_localization_source = pose_source;
        initial_odometry_position = localization_pose.position.head<2>();
        initial_odometry_heading = localization_pose.heading;
        goal_initialized = true;
        spdlog::info(
            "Navigation goal initialized from {} pose "
            "(x={:.3f}, y={:.3f}, yaw={:.3f}) "
            "goal=(x={:.3f}, y={:.3f}, yaw={:.3f})",
            active_localization_source,
            initial_odometry_position.x(),
            initial_odometry_position.y(),
            initial_odometry_heading,
            goal.position.x(),
            goal.position.y(),
            goal.yaw
        );
    }

    const Eigen::Vector2f displacement =
        localization_pose.position.head<2>() - initial_odometry_position;
    const float cos_initial_heading = std::cos(initial_odometry_heading);
    const float sin_initial_heading = std::sin(initial_odometry_heading);
    // Express GLIM displacement in the body-aligned frame captured on entry.
    estimated_position = Eigen::Vector2f(
        cos_initial_heading * displacement.x() +
            sin_initial_heading * displacement.y(),
        -sin_initial_heading * displacement.x() +
            cos_initial_heading * displacement.y()
    );
    const float heading = wrap_to_pi(
        localization_pose.heading - initial_odometry_heading
    );

    const Eigen::Vector2f target_position(goal.position.x(), goal.position.y());
    const Eigen::Vector2f delta_navigation =
        target_position - estimated_position;
    const float cos_heading = std::cos(heading);
    const float sin_heading = std::sin(heading);
    const Eigen::Vector2f delta_body(
        cos_heading * delta_navigation.x() +
            sin_heading * delta_navigation.y(),
        -sin_heading * delta_navigation.x() +
            cos_heading * delta_navigation.y()
    );
    const float distance = delta_body.norm();
    const float heading_error = wrap_to_pi(goal.yaw - heading);

    command[0] = std::clamp(
        position_control_stiffness * delta_body.x(),
        lin_vel_x_range[0],
        lin_vel_x_range[1]
    );
    command[1] = std::clamp(
        position_control_stiffness * delta_body.y(),
        lin_vel_y_range[0],
        lin_vel_y_range[1]
    );
    command[2] = std::clamp(
        heading_control_stiffness * heading_error,
        ang_vel_z_range[0],
        ang_vel_z_range[1]
    );

    const bool reached_position_now = distance < position_tolerance;
    const bool reached_heading_now = std::abs(heading_error) < heading_tolerance;
    const float linear_speed = std::hypot(command[0], command[1]);
    if (!reached_position_now && linear_speed > 1.0e-6f && linear_speed < min_approach_speed) {
        const float speed_scale = min_approach_speed / linear_speed;
        command[0] *= speed_scale;
        command[1] *= speed_scale;
    }

    position_reached = position_reached || reached_position_now;
    goal_reached = goal_reached || (position_reached && reached_heading_now);
    if (position_reached) {
        command[0] = 0.0f;
        command[1] = 0.0f;
    }
    if (reached_heading_now) {
        command[2] = 0.0f;
    }
    if (goal_reached) {
        command = {0.0f, 0.0f, 0.0f};
    }

    const float camera_heading_in_body = wrap_to_pi(
        camera_heading() - root_heading()
    );
    update_marker_observation(wrap_to_pi(heading + camera_heading_in_body));
    write_pose_log(
        true,
        pose_source,
        localization_pose,
        heading,
        delta_body,
        heading_error,
        distance
    );
}

void State_Navigation::update_marker_observation(float camera_heading_relative)
{
    const Eigen::Vector2f stand_off(
        stand_off_distance * std::cos(goal.yaw),
        stand_off_distance * std::sin(goal.yaw)
    );

    const Eigen::Vector2f object_position =
        Eigen::Vector2f(goal.position.x(), goal.position.y()) + stand_off;
    const Eigen::Vector2f camera_offset_navigation(
        std::cos(camera_heading_relative) * camera_forward_offset,
        std::sin(camera_heading_relative) * camera_forward_offset
    );
    const Eigen::Vector2f camera_position =
        estimated_position + camera_offset_navigation;
    const Eigen::Vector2f marker_delta_navigation =
        object_position - camera_position;
    const float cos_heading = std::cos(camera_heading_relative);
    const float sin_heading = std::sin(camera_heading_relative);
    const float marker_x =
        cos_heading * marker_delta_navigation.x() +
        sin_heading * marker_delta_navigation.y();
    const float marker_y =
        -sin_heading * marker_delta_navigation.x() +
        cos_heading * marker_delta_navigation.y();
    const float marker_heading_error = wrap_to_pi(
        static_cast<float>(M_PI) - camera_heading_relative
    );

    marker_pose_camera = {
        marker_x,
        marker_y,
        marker_camera_height_offset,
        std::sqrt(
            marker_x * marker_x +
            marker_y * marker_y +
            marker_camera_height_offset * marker_camera_height_offset
        ),
        std::sin(marker_heading_error),
        std::cos(marker_heading_error),
    };
}

void State_Navigation::set_safe_stand_observations()
{
    const float marker_x = stand_off_distance - camera_forward_offset;
    marker_pose_camera = {
        marker_x,
        0.0f,
        marker_camera_height_offset,
        std::sqrt(
            marker_x * marker_x +
            marker_camera_height_offset * marker_camera_height_offset
        ),
        0.0f,
        -1.0f,
    };
}

void State_Navigation::open_pose_log()
{
    close_pose_log();
    if (!pose_log_enabled) {
        return;
    }

    auto resolved_path = pose_log_path;
    if (resolved_path.is_relative()) {
        resolved_path = param::proj_dir / resolved_path;
    }
    std::error_code error;
    std::filesystem::create_directories(resolved_path.parent_path(), error);
    if (error) {
        spdlog::error(
            "Failed to create Navigation pose log directory {}: {}",
            resolved_path.parent_path().string(),
            error.message()
        );
        return;
    }

    pose_log.open(resolved_path, std::ios::out | std::ios::trunc);
    if (!pose_log.is_open()) {
        spdlog::error(
            "Failed to open Navigation pose log {}.",
            resolved_path.string()
        );
        return;
    }

    pose_log
        << "time_s,source,localization_available,robot_x_m,robot_y_m,"
        << "robot_yaw_rad,slam_pose_x_m,slam_pose_y_m,slam_pose_z_m,"
        << "slam_pose_qx,slam_pose_qy,slam_pose_qz,slam_pose_qw,"
        << "slam_pose_theta_rad,goal_x_m,goal_y_m,goal_yaw_rad,"
        << "command_rel_pos_x_m,command_rel_pos_y_m,command_rel_yaw_rad,"
        << "goal_body_x_m,goal_body_y_m,goal_yaw_error_rad,position_error_m,"
        << "command_vx_mps,command_vy_mps,command_wz_radps,"
        << "cmd_vx_mps,cmd_vy_mps,cmd_wz_radps,"
        << "projected_gravity_x,projected_gravity_y,projected_gravity_z,"
        << "goal_reached";
    const std::size_t joint_count = env->robot->data.joint_ids_map.size();
    for (std::size_t i = 0; i < joint_count; ++i) {
        pose_log << ",policy_action_joint_" << std::setw(2) << std::setfill('0') << i;
    }
    for (std::size_t i = 0; i < joint_count; ++i) {
        pose_log << ",command_joint_" << std::setw(2) << std::setfill('0') << i;
    }
    for (std::size_t i = 0; i < joint_count; ++i) {
        pose_log << ",encoder_joint_" << std::setw(2) << std::setfill('0') << i;
    }
    for (std::size_t i = 0; i < joint_count; ++i) {
        pose_log << ",action_joint_" << std::setw(2) << std::setfill('0') << i;
    }
    pose_log << std::setfill(' ') << '\n';
    pose_log.flush();
    navigation_started_at = std::chrono::steady_clock::now();
    pose_log_sample_count = 0;
    spdlog::info("Navigation pose log: {}", resolved_path.string());
}

void State_Navigation::close_pose_log()
{
    if (pose_log.is_open()) {
        pose_log.flush();
        pose_log.close();
    }
}

void State_Navigation::write_pose_log(
    bool localization_available,
    const std::string& source,
    const LocalizationPose& localization_pose,
    float robot_heading,
    const Eigen::Vector2f& goal_position_body,
    float goal_heading_error,
    float position_error)
{
    if (!pose_log.is_open()) {
        return;
    }

    const float elapsed_seconds = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - navigation_started_at
    ).count();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto& slam_position = localization_pose.position;
    const auto& slam_orientation = localization_pose.orientation;
    const auto encoder_joints = env->robot->data.joint_pos;
    const auto projected_gravity = env->robot->data.projected_gravity_b;
    std::vector<float> policy_action;
    std::vector<float> command_joints;
    std::vector<float> action_joints;
    {
        std::lock_guard<std::mutex> lock(joint_command_mutex);
        policy_action = latest_policy_action;
        command_joints = latest_command_joints;
        action_joints = last_sent_action_joints;
    }
    pose_log
        << std::fixed << std::setprecision(6)
        << elapsed_seconds << ','
        << source << ','
        << (localization_available ? 1 : 0) << ','
        << (localization_available ? estimated_position.x() :
            nan) << ','
        << (localization_available ? estimated_position.y() :
            nan) << ','
        << robot_heading << ','
        << (localization_available ? slam_position.x() : nan) << ','
        << (localization_available ? slam_position.y() : nan) << ','
        << (localization_available ? slam_position.z() : nan) << ','
        << (localization_available ? slam_orientation.x() : nan) << ','
        << (localization_available ? slam_orientation.y() : nan) << ','
        << (localization_available ? slam_orientation.z() : nan) << ','
        << (localization_available ? slam_orientation.w() : nan) << ','
        << (localization_available ? localization_pose.heading : nan) << ','
        << goal.position.x() << ','
        << goal.position.y() << ','
        << goal.yaw << ','
        << goal_position_body.x() << ','
        << goal_position_body.y() << ','
        << goal_heading_error << ','
        << goal_position_body.x() << ','
        << goal_position_body.y() << ','
        << goal_heading_error << ','
        << position_error << ','
        << command[0] << ','
        << command[1] << ','
        << command[2] << ','
        << command[0] << ','
        << command[1] << ','
        << command[2] << ','
        << projected_gravity.x() << ','
        << projected_gravity.y() << ','
        << projected_gravity.z() << ','
        << (goal_reached ? 1 : 0);

    const std::size_t joint_count = env->robot->data.joint_ids_map.size();
    const auto write_joint_values = [&](const auto& values) {
        for (std::size_t i = 0; i < joint_count; ++i) {
            pose_log << ',' << (i < static_cast<std::size_t>(values.size())
                ? values[i]
                : nan);
        }
    };
    write_joint_values(policy_action);
    write_joint_values(command_joints);
    write_joint_values(encoder_joints);
    write_joint_values(action_joints);
    pose_log << '\n';

    ++pose_log_sample_count;
    if (pose_log_sample_count % pose_log_flush_interval == 0) {
        pose_log.flush();
    }
}

float State_Navigation::root_heading() const
{
    return quaternion_heading(env->robot->data.root_quat_w);
}

float State_Navigation::camera_heading() const
{
    const auto& joint_pos = env->robot->data.joint_pos;
    const Eigen::Quaternionf camera_quat =
        env->robot->data.root_quat_w *
        Eigen::AngleAxisf(joint_pos[12], Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(joint_pos[13], Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(joint_pos[14], Eigen::Vector3f::UnitY());
    return quaternion_heading(camera_quat);
}

float State_Navigation::wrap_to_pi(float angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}
