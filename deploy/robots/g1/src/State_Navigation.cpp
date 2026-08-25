#include "State_Navigation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
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

std::array<int, 2> read_int_range(
    const YAML::Node& node,
    const std::array<int, 2>& fallback)
{
    if (!node || !node.IsSequence() || node.size() != 2) {
        return fallback;
    }
    return {node[0].as<int>(), node[1].as<int>()};
}

float smoothstep(float ratio)
{
    ratio = std::clamp(ratio, 0.0f, 1.0f);
    return ratio * ratio * (3.0f - 2.0f * ratio);
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

REGISTER_OBSERVATION(navigation_future_path_poses)
{
    (void)env;
    (void)params;
    return State_Navigation::instance()->future_path_observation();
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
    target_path_log_path = nav_cfg["target_path_log_path"].as<std::string>(
        target_path_log_path.string()
    );
    pose_log_flush_interval = nav_cfg["pose_log_flush_interval"].as<std::size_t>(
        pose_log_flush_interval
    );
    odometry_timeout = nav_cfg["odometry_timeout"].as<float>(odometry_timeout);
    const auto configured_trajectory = nav_cfg["trajectory"];
    const YAML::Node trajectory_cfg = configured_trajectory
        ? configured_trajectory
        : YAML::Node(YAML::NodeType::Map);
    reference_times = read_range(
        trajectory_cfg["reference_times"], reference_times
    );
    motion_duration = trajectory_cfg["motion_duration"].as<float>(
        motion_duration
    );
    stop_hold_duration = trajectory_cfg["stop_hold_duration"].as<float>(
        stop_hold_duration
    );
    start_ramp_duration = trajectory_cfg["start_ramp_duration"].as<float>(
        start_ramp_duration
    );
    stop_ramp_duration = trajectory_cfg["stop_ramp_duration"].as<float>(
        stop_ramp_duration
    );
    num_segments_range = read_int_range(
        trajectory_cfg["num_segments_range"], num_segments_range
    );
    min_segment_duration = trajectory_cfg["min_segment_duration"].as<float>(
        min_segment_duration
    );
    min_radius = trajectory_cfg["min_radius"].as<float>(min_radius);
    straight_probability = trajectory_cfg["straight_probability"].as<float>(
        straight_probability
    );
    curvature_exponent = trajectory_cfg["curvature_exponent"].as<float>(
        curvature_exponent
    );
    se2_speed_range = read_range(
        trajectory_cfg["se2_speed_range"], se2_speed_range
    );
    characteristic_length = trajectory_cfg["characteristic_length"].as<float>(
        characteristic_length
    );
    tracking_gain = trajectory_cfg["tracking_gain"].as<float>(tracking_gain);
    max_linear_speed = trajectory_cfg["max_linear_speed"].as<float>(
        max_linear_speed
    );
    max_angular_speed = trajectory_cfg["max_angular_speed"].as<float>(
        max_angular_speed
    );
    standing_probability = trajectory_cfg["standing_probability"].as<float>(
        standing_probability
    );
    trajectory_seed = trajectory_cfg["random_seed"].as<unsigned int>(
        trajectory_seed
    );
    trajectory_step_dt = deploy_cfg["step_dt"].as<float>(trajectory_step_dt);

    // These values are only used when an older marker-observation policy is
    // selected during migration to the 120-input trajectory actor.
    stand_off_distance = nav_cfg["stand_off_distance"].as<float>(stand_off_distance);
    camera_forward_offset = nav_cfg["camera_forward_offset"].as<float>(camera_forward_offset);
    marker_camera_height_offset = nav_cfg["marker_camera_height_offset"].as<float>(marker_camera_height_offset);
    position_tolerance = nav_cfg["position_tolerance"].as<float>(position_tolerance);
    heading_tolerance = nav_cfg["heading_tolerance"].as<float>(heading_tolerance);

    if (trajectory_seed == 0U) {
        std::random_device random_device;
        trajectory_rng.seed(random_device());
    } else {
        trajectory_rng.seed(trajectory_seed);
    }

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
        (pose_log_path.empty() || target_path_log_path.empty() ||
         pose_log_flush_interval == 0)) {
        throw std::runtime_error("Invalid Navigation pose log configuration.");
    }
    if (!std::isfinite(trajectory_step_dt) || trajectory_step_dt <= 0.0f ||
        !std::isfinite(motion_duration) || motion_duration <= 0.0f ||
        !std::isfinite(stop_hold_duration) || stop_hold_duration < 0.0f ||
        !std::isfinite(min_segment_duration) || min_segment_duration <= 0.0f) {
        throw std::runtime_error(
            "Navigation trajectory durations must be valid and finite."
        );
    }
    const auto is_step_aligned = [this](float duration) {
        const float steps = duration / trajectory_step_dt;
        return std::isfinite(duration) &&
            std::abs(steps - std::round(steps)) < 1.0e-4f;
    };
    const float trajectory_duration = motion_duration + stop_hold_duration;
    const int motion_steps = static_cast<int>(std::lround(
        motion_duration / trajectory_step_dt
    ));
    const int minimum_segment_steps = static_cast<int>(std::ceil(
        min_segment_duration / trajectory_step_dt
    ));
    const bool valid_trajectory =
        motion_duration > 0.0f &&
        stop_hold_duration >= 0.0f &&
        is_step_aligned(motion_duration) &&
        is_step_aligned(trajectory_duration) &&
        start_ramp_duration >= 0.0f &&
        stop_ramp_duration >= 0.0f &&
        start_ramp_duration + stop_ramp_duration <= motion_duration &&
        reference_times[0] > 0.0f &&
        reference_times[0] < reference_times[1] &&
        reference_times[1] <= trajectory_duration &&
        num_segments_range[0] >= 1 &&
        num_segments_range[0] <= num_segments_range[1] &&
        num_segments_range[1] * minimum_segment_steps <= motion_steps &&
        min_segment_duration > 0.0f &&
        min_radius > 0.0f &&
        straight_probability >= 0.0f && straight_probability <= 1.0f &&
        curvature_exponent > 0.0f &&
        se2_speed_range[0] > 0.0f &&
        se2_speed_range[0] <= se2_speed_range[1] &&
        characteristic_length > 0.0f && tracking_gain >= 0.0f &&
        se2_speed_range[1] <= max_linear_speed &&
        se2_speed_range[1] / characteristic_length <= max_angular_speed &&
        standing_probability >= 0.0f && standing_probability <= 1.0f &&
        position_tolerance > 0.0f && heading_tolerance > 0.0f;
    if (!valid_trajectory) {
        throw std::runtime_error("Invalid Navigation trajectory configuration.");
    }
    set_safe_stand_observations();

    auto articulation = std::make_shared<
        unitree::BaseArticulation<LowState_t::SharedPtr>
    >(FSMState::lowstate);
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(deploy_cfg, articulation);
    env->alg = std::make_unique<isaaclab::OrtRunner>(
        policy_dir / "exported" / "policy.onnx"
    );

    const auto arm_pose_cfg = deploy_cfg["commands"]["arm_pose"];
    if (arm_pose_cfg && arm_pose_cfg["poses"]) {
        const std::string default_pose =
            arm_pose_cfg["default_pose"].as<std::string>("down");
        const auto default_target = arm_pose_cfg["poses"][default_pose]
            .as<std::vector<float>>();
        if (arm_pose_cfg["joint_ids"]) {
            arm_command_joint_ids =
                arm_pose_cfg["joint_ids"].as<std::vector<int>>();
        } else if (default_target.size() <= env->robot->data.joint_ids_map.size()) {
            // G1 arm joints occupy the trailing policy-order entries. This
            // fallback keeps older arm policy configs usable, while an
            // explicit joint_ids list remains preferable.
            const int first_arm_joint = static_cast<int>(
                env->robot->data.joint_ids_map.size() - default_target.size()
            );
            for (std::size_t i = 0; i < default_target.size(); ++i) {
                arm_command_joint_ids.push_back(
                    first_arm_joint + static_cast<int>(i)
                );
            }
        }

        if (arm_command_joint_ids.size() != default_target.size()) {
            throw std::runtime_error(
                "arm_pose joint_ids and pose targets must have the same dimension."
            );
        }
        std::vector<bool> used_joint_ids(
            env->robot->data.joint_ids_map.size(), false
        );
        for (const int joint_id : arm_command_joint_ids) {
            if (joint_id < 0 ||
                joint_id >= static_cast<int>(used_joint_ids.size()) ||
                used_joint_ids[joint_id]) {
                throw std::runtime_error(
                    "arm_pose joint_ids must be unique valid policy-order indices."
                );
            }
            used_joint_ids[joint_id] = true;
        }
        for (const auto& pose : arm_pose_cfg["poses"]) {
            if (pose.second.as<std::vector<float>>().size() !=
                arm_command_joint_ids.size()) {
                throw std::runtime_error(
                    "Every arm_pose target must match arm_pose joint_ids."
                );
            }
        }
        spdlog::info(
            "Navigation command-joint telemetry: {} arm joints.",
            arm_command_joint_ids.size()
        );
    }

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
                latest_processed_action_joints =
                    env->action_manager->processed_actions();
            }
            action_ready = true;

            std::this_thread::sleep_until(sleep_until);
            sleep_until += dt;
        }
    });
}

bool State_Navigation::prepare_enter()
{
    return true;
}

void State_Navigation::cancel_prepare_enter()
{
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
    const auto& action = latest_processed_action_joints;
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

std::vector<float> State_Navigation::future_path_observation() const
{
    return std::vector<float>(future_path_poses.begin(), future_path_poses.end());
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
    trajectory_initialized = false;
    active_localization_source.clear();
    localization_warning_reported = false;
    goal_reached = false;
    goal = TrajectoryPose{};
    trajectory_poses.clear();
    trajectory_progresses.clear();
    trajectory_linear_velocities.clear();
    trajectory_angular_velocities.clear();
    closest_progress = 0.0f;
    command = {0.0f, 0.0f, 0.0f};
    {
        std::lock_guard<std::mutex> lock(joint_command_mutex);
        latest_policy_action.assign(
            env->robot->data.joint_ids_map.size(),
            std::numeric_limits<float>::quiet_NaN()
        );
        latest_processed_action_joints.assign(
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

void State_Navigation::generate_trajectory()
{
    const int motion_steps = static_cast<int>(std::lround(
        motion_duration / trajectory_step_dt
    ));
    const float trajectory_duration = motion_duration + stop_hold_duration;
    const int episode_steps = static_cast<int>(std::lround(
        trajectory_duration / trajectory_step_dt
    ));
    const int minimum_segment_steps = static_cast<int>(std::ceil(
        min_segment_duration / trajectory_step_dt
    ));

    std::uniform_int_distribution<int> segment_count_distribution(
        num_segments_range[0], num_segments_range[1]
    );
    const int segment_count = segment_count_distribution(trajectory_rng);
    std::vector<int> segment_steps(segment_count, minimum_segment_steps);
    int extra_steps = motion_steps - segment_count * minimum_segment_steps;
    std::uniform_int_distribution<int> segment_distribution(0, segment_count - 1);
    while (extra_steps-- > 0) {
        ++segment_steps[segment_distribution(trajectory_rng)];
    }
    std::vector<int> segment_ends(segment_count);
    std::partial_sum(
        segment_steps.begin(), segment_steps.end(), segment_ends.begin()
    );

    std::uniform_real_distribution<float> unit_distribution(0.0f, 1.0f);
    std::vector<float> curvatures(segment_count, 0.0f);
    for (float& curvature : curvatures) {
        if (unit_distribution(trajectory_rng) < straight_probability) {
            continue;
        }
        const float magnitude = std::pow(
            unit_distribution(trajectory_rng), curvature_exponent
        ) / min_radius;
        curvature = unit_distribution(trajectory_rng) < 0.5f
            ? -magnitude
            : magnitude;
    }

    std::uniform_real_distribution<float> speed_distribution(
        se2_speed_range[0], se2_speed_range[1]
    );
    const bool standing =
        unit_distribution(trajectory_rng) < standing_probability;
    const float se2_speed = standing ? 0.0f : speed_distribution(trajectory_rng);

    trajectory_poses.assign(episode_steps + 1, TrajectoryPose{});
    trajectory_progresses.assign(episode_steps + 1, 0.0f);
    trajectory_linear_velocities.assign(episode_steps, 0.0f);
    trajectory_angular_velocities.assign(episode_steps, 0.0f);

    int segment_index = 0;
    float accumulated_rotation = 0.0f;
    for (int step = 0; step < motion_steps; ++step) {
        while (step >= segment_ends[segment_index]) {
            ++segment_index;
        }
        const float curvature = curvatures[segment_index];
        const float midpoint_time = (static_cast<float>(step) + 0.5f)
            * trajectory_step_dt;
        float speed_scale = 1.0f;
        if (midpoint_time < start_ramp_duration) {
            speed_scale = smoothstep(midpoint_time / start_ramp_duration);
        }
        const float stop_start = motion_duration - stop_ramp_duration;
        if (midpoint_time > stop_start) {
            speed_scale = smoothstep(
                (motion_duration - midpoint_time) / stop_ramp_duration
            );
        }
        const float linear_velocity = speed_scale * se2_speed / std::sqrt(
            1.0f + std::pow(characteristic_length * curvature, 2.0f)
        );
        const float angular_velocity = curvature * linear_velocity;
        trajectory_linear_velocities[step] = linear_velocity;
        trajectory_angular_velocities[step] = angular_velocity;
        accumulated_rotation += std::abs(angular_velocity) * trajectory_step_dt;

        const float distance = linear_velocity * trajectory_step_dt;
        const float angle = curvature * distance;
        const auto sinc = [](float value) {
            if (std::abs(value) < 1.0e-6f) {
                return 1.0f - value * value / 6.0f;
            }
            return std::sin(value) / value;
        };
        const float local_x = distance * sinc(angle);
        const float half_angle_sinc = sinc(0.5f * angle);
        const float local_y = 0.5f * distance * angle
            * half_angle_sinc * half_angle_sinc;
        const TrajectoryPose& current = trajectory_poses[step];
        TrajectoryPose& next = trajectory_poses[step + 1];
        const float cos_heading = std::cos(current.yaw);
        const float sin_heading = std::sin(current.yaw);
        next.position.x() = current.position.x()
            + cos_heading * local_x - sin_heading * local_y;
        next.position.y() = current.position.y()
            + sin_heading * local_x + cos_heading * local_y;
        next.yaw = current.yaw + angle;
        trajectory_progresses[step + 1] =
            trajectory_progresses[step] + distance;
    }
    if (accumulated_rotation >= 2.0f * static_cast<float>(M_PI)) {
        throw std::runtime_error(
            "Generated Navigation trajectory accumulates a full turn."
        );
    }
    for (int step = motion_steps + 1; step <= episode_steps; ++step) {
        trajectory_poses[step] = trajectory_poses[motion_steps];
        trajectory_progresses[step] = trajectory_progresses[motion_steps];
    }

    goal = trajectory_poses[motion_steps];
    trajectory_started_at = std::chrono::steady_clock::now();
    write_target_path_log();
    spdlog::info(
        "Navigation path generated: {} segments, SE(2) speed={:.3f}, "
        "goal=(x={:.3f}, y={:.3f}, yaw={:.3f}).",
        segment_count,
        se2_speed,
        goal.position.x(),
        goal.position.y(),
        goal.yaw
    );
}

std::size_t State_Navigation::closest_trajectory_index(
    const Eigen::Vector2f& position) const
{
    if (trajectory_poses.empty()) {
        return 0;
    }
    const std::size_t motion_steps = static_cast<std::size_t>(std::lround(
        motion_duration / trajectory_step_dt
    ));
    const std::size_t last_path_index = std::min(
        motion_steps, trajectory_poses.size() - 1
    );
    std::size_t closest_index = 0;
    float closest_distance_sq = std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index <= last_path_index; ++index) {
        const float distance_sq =
            (trajectory_poses[index].position - position).squaredNorm();
        if (distance_sq < closest_distance_sq) {
            closest_distance_sq = distance_sq;
            closest_index = index;
        }
    }
    return closest_index;
}

std::size_t State_Navigation::offset_trajectory_index(
    std::size_t reference_index,
    float offset_seconds) const
{
    if (trajectory_poses.empty()) {
        return 0;
    }
    const auto offset_steps = static_cast<std::size_t>(std::max(
        0L, std::lround(offset_seconds / trajectory_step_dt)
    ));
    const std::size_t index = reference_index + offset_steps;
    return std::min(index, trajectory_poses.size() - 1);
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

    if (!trajectory_initialized) {
        active_localization_source = pose_source;
        initial_odometry_position = localization_pose.position.head<2>();
        initial_odometry_heading = localization_pose.heading;
        trajectory_initialized = true;
        generate_trajectory();
        spdlog::info(
            "Navigation path frame initialized from {} pose "
            "(x={:.3f}, y={:.3f}, yaw={:.3f}).",
            active_localization_source,
            initial_odometry_position.x(),
            initial_odometry_position.y(),
            initial_odometry_heading
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

    const std::size_t reference_index = closest_trajectory_index(
        estimated_position
    );
    closest_progress = trajectory_progresses[reference_index];
    const float startup_elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - trajectory_started_at
    ).count();
    const std::size_t startup_ramp_steps = static_cast<std::size_t>(std::lround(
        start_ramp_duration / trajectory_step_dt
    ));
    const std::size_t startup_step = std::min(
        static_cast<std::size_t>(std::max(
            0L, std::lround(startup_elapsed / trajectory_step_dt)
        )),
        startup_ramp_steps
    );
    const std::size_t velocity_index = std::min(
        std::max(reference_index, startup_step),
        trajectory_linear_velocities.size() - 1
    );
    const TrajectoryPose& reference = trajectory_poses[reference_index];
    const Eigen::Vector2f position_error_navigation =
        reference.position - estimated_position;
    const float reference_linear_velocity =
        trajectory_linear_velocities[velocity_index];
    const float reference_angular_velocity =
        trajectory_angular_velocities[velocity_index];
    const Eigen::Vector2f reference_velocity_navigation(
        reference_linear_velocity * std::cos(reference.yaw),
        reference_linear_velocity * std::sin(reference.yaw)
    );
    const Eigen::Vector2f velocity_navigation =
        reference_velocity_navigation
        + tracking_gain * position_error_navigation;
    const float cos_heading = std::cos(heading);
    const float sin_heading = std::sin(heading);
    const Eigen::Vector2f position_error_body(
        cos_heading * position_error_navigation.x() +
            sin_heading * position_error_navigation.y(),
        -sin_heading * position_error_navigation.x() +
            cos_heading * position_error_navigation.y()
    );
    command[0] =
        cos_heading * velocity_navigation.x()
        + sin_heading * velocity_navigation.y();
    command[1] =
        -sin_heading * velocity_navigation.x()
        + cos_heading * velocity_navigation.y();
    const float linear_speed = std::hypot(command[0], command[1]);
    if (linear_speed > max_linear_speed) {
        const float speed_scale = max_linear_speed / linear_speed;
        command[0] *= speed_scale;
        command[1] *= speed_scale;
    }
    const float reference_heading_error = wrap_to_pi(
        reference.yaw - heading
    );
    command[2] = std::clamp(
        reference_angular_velocity + tracking_gain * reference_heading_error,
        -max_angular_speed,
        max_angular_speed
    );

    const float goal_position_error =
        (goal.position - estimated_position).norm();
    const float goal_heading_error = wrap_to_pi(goal.yaw - heading);
    goal_reached =
        goal_position_error < position_tolerance &&
        std::abs(goal_heading_error) < heading_tolerance;

    const float camera_heading_in_body = wrap_to_pi(
        camera_heading() - root_heading()
    );
    update_marker_observation(wrap_to_pi(heading + camera_heading_in_body));
    update_future_path_observation(heading, reference_index);
    write_pose_log(
        true,
        pose_source,
        localization_pose,
        heading,
        position_error_body,
        reference_heading_error,
        position_error_navigation.norm(),
        closest_progress
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
        goal.yaw + static_cast<float>(M_PI) - camera_heading_relative
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

void State_Navigation::update_future_path_observation(
    float robot_heading,
    std::size_t closest_reference_index)
{
    const float cos_heading = std::cos(robot_heading);
    const float sin_heading = std::sin(robot_heading);
    for (std::size_t reference_index = 0;
         reference_index < reference_times.size();
        ++reference_index) {
        const TrajectoryPose& future = trajectory_poses[
            offset_trajectory_index(
                closest_reference_index,
                reference_times[reference_index]
            )
        ];
        const Eigen::Vector2f delta_navigation =
            future.position - estimated_position;
        const std::size_t offset = 4 * reference_index;
        future_path_poses[offset] =
            cos_heading * delta_navigation.x()
            + sin_heading * delta_navigation.y();
        future_path_poses[offset + 1] =
            -sin_heading * delta_navigation.x()
            + cos_heading * delta_navigation.y();
        const float heading_error = wrap_to_pi(future.yaw - robot_heading);
        future_path_poses[offset + 2] = std::cos(heading_error);
        future_path_poses[offset + 3] = std::sin(heading_error);
    }
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
    future_path_poses = {
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };
}

void State_Navigation::write_target_path_log() const
{
    if (!pose_log_enabled || trajectory_poses.empty()) {
        return;
    }
    auto resolved_path = target_path_log_path;
    if (resolved_path.is_relative()) {
        resolved_path = param::proj_dir / resolved_path;
    }
    std::error_code error;
    std::filesystem::create_directories(resolved_path.parent_path(), error);
    if (error) {
        spdlog::error(
            "Failed to create Navigation target path directory {}: {}",
            resolved_path.parent_path().string(),
            error.message()
        );
        return;
    }
    std::ofstream target_path_log(resolved_path, std::ios::out | std::ios::trunc);
    if (!target_path_log.is_open()) {
        spdlog::error(
            "Failed to open Navigation target path log {}.",
            resolved_path.string()
        );
        return;
    }
    const std::size_t motion_steps = static_cast<std::size_t>(std::lround(
        motion_duration / trajectory_step_dt
    ));
    target_path_log << "time_s,x_m,y_m,yaw_rad,is_goal\n";
    target_path_log << std::fixed << std::setprecision(6);
    for (std::size_t index = 0;
         index <= motion_steps && index < trajectory_poses.size();
         ++index) {
        const TrajectoryPose& pose = trajectory_poses[index];
        target_path_log
            << index * trajectory_step_dt << ','
            << pose.position.x() << ','
            << pose.position.y() << ','
            << pose.yaw << ','
            << (index == motion_steps ? 1 : 0) << '\n';
    }
    target_path_log.flush();
    spdlog::info("Navigation target path log: {}", resolved_path.string());
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

    // Remove the previous run's path immediately. The generated path is
    // written once the first fresh localization sample defines its frame.
    auto resolved_target_path = target_path_log_path;
    if (resolved_target_path.is_relative()) {
        resolved_target_path = param::proj_dir / resolved_target_path;
    }
    error.clear();
    std::filesystem::create_directories(
        resolved_target_path.parent_path(), error
    );
    if (error) {
        spdlog::error(
            "Failed to create Navigation target path directory {}: {}",
            resolved_target_path.parent_path().string(),
            error.message()
        );
    } else {
        std::ofstream target_path_log(
            resolved_target_path, std::ios::out | std::ios::trunc
        );
        if (target_path_log.is_open()) {
            target_path_log << "time_s,x_m,y_m,yaw_rad,is_goal\n";
        } else {
            spdlog::error(
                "Failed to clear Navigation target path log {}.",
                resolved_target_path.string()
            );
        }
    }

    pose_log
        << "time_s,source,localization_available,robot_x_m,robot_y_m,"
        << "robot_yaw_rad,slam_pose_x_m,slam_pose_y_m,slam_pose_z_m,"
        << "slam_pose_qx,slam_pose_qy,slam_pose_qz,slam_pose_qw,"
        << "slam_pose_theta_rad,goal_x_m,goal_y_m,goal_yaw_rad,"
        << "command_rel_pos_x_m,command_rel_pos_y_m,command_rel_yaw_rad,"
        << "goal_body_x_m,goal_body_y_m,goal_yaw_error_rad,position_error_m,"
        << "closest_progress_m,"
        << "command_vx_mps,command_vy_mps,command_wz_radps,"
        << "cmd_vx_mps,cmd_vy_mps,cmd_wz_radps,"
        << "projected_gravity_x,projected_gravity_y,projected_gravity_z,"
        << "goal_reached";
    const std::size_t joint_count = env->robot->data.joint_ids_map.size();
    for (std::size_t i = 0; i < joint_count; ++i) {
        pose_log << ",policy_action_joint_" << std::setw(2) << std::setfill('0') << i;
    }
    for (const int joint_id : arm_command_joint_ids) {
        pose_log
            << ",command_joint_"
            << std::setw(2) << std::setfill('0') << joint_id;
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
    const Eigen::Vector2f& reference_position_body,
    float reference_heading_error,
    float position_error,
    float closest_progress_value)
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
    // command_joints is the arm-pose target selected by the deploy input
    // (L1 + Up/Down), not the policy's processed full-body action.
    const auto command_joints = env->get_command("arm_pose_target");
    std::vector<float> action_joints;
    {
        std::lock_guard<std::mutex> lock(joint_command_mutex);
        policy_action = latest_policy_action;
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
        << reference_position_body.x() << ','
        << reference_position_body.y() << ','
        << reference_heading_error << ','
        << reference_position_body.x() << ','
        << reference_position_body.y() << ','
        << reference_heading_error << ','
        << position_error << ','
        << closest_progress_value << ','
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
    for (std::size_t i = 0; i < arm_command_joint_ids.size(); ++i) {
        pose_log << ',' << (i < command_joints.size() ? command_joints[i] : nan);
    }
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
