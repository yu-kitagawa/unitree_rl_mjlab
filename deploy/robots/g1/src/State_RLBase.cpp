#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}

REGISTER_OBSERVATION(arm_pose_commands)
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

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    const auto arm_pose_cfg = env->cfg["commands"]["arm_pose"];
    if (arm_pose_cfg && arm_pose_cfg["poses"]) {
        const auto & joystick = FSMState::lowstate->joystick;
        std::string requested_pose;

        if (joystick.LB.pressed && joystick.up.on_pressed) {
            requested_pose = "up";
        } else if (joystick.LB.pressed && joystick.down.on_pressed) {
            requested_pose = "down";
        }

        if (!requested_pose.empty()) {
            env->set_command(
                "arm_pose_target",
                arm_pose_cfg["poses"][requested_pose].as<std::vector<float>>()
            );
            spdlog::info(
                "Arm pose target: {} (max joint speed: {:.2f} rad/s)",
                requested_pose,
                arm_pose_cfg["max_joint_speed"].as<float>(0.15f)
            );
        }
    }

    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
