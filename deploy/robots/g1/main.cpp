#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "State_Navigation.h"

#ifdef G1_NAVIGATION_WITH_ROS2
#include <rclcpp/rclcpp.hpp>

#include <vector>
#endif

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
#ifdef G1_NAVIGATION_WITH_ROS2
    // Keep ROS-specific arguments away from the controller's Boost parser.
    auto controller_arguments = rclcpp::remove_ros_arguments(argc, argv);
    std::vector<char*> controller_argv;
    controller_argv.reserve(controller_arguments.size());
    for (auto& argument : controller_arguments) {
        controller_argv.push_back(argument.data());
    }

    // Load parameters
    auto vm = param::helper(
        static_cast<int>(controller_argv.size()),
        controller_argv.data()
    );

    // Preserve the controller's existing process-level signal handling.
    rclcpp::init(
        argc,
        argv,
        rclcpp::InitOptions(),
        rclcpp::SignalHandlerOptions::None
    );
#else
    auto vm = param::helper(argc, argv);
#endif

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + A] to start controlling the robot.\n";
    std::cout << "Press [R2 + B], then enter Navigation goal x y yaw_rad in this terminal.\n";
    std::cout << "Press [R2 + A] to return from Navigation to Velocity.\n";
    std::cout << "In arm-enabled Navigation, press [L1 + Up/Down] to raise/lower the arms.\n";
    std::cout << "And then press [R1 + A/B/Y/X] to control the robot dance.\n";

    while (true)
    {
        State_Navigation::instance()->process_goal_input();
        usleep(10000);
    }
    
    return 0;
}
