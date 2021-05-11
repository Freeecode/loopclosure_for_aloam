/**
 * @brief loop closure detection for aloam
 * 
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

//ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include "Scancontext/Scancontext.h"

// GTSAM
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

ros::Publisher  pub_path;

// Buffer
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::queue<nav_msgs::OdometryConstPtr> odometryBuf;
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> laserCloudBuf;
std::vector<std_msgs::Header> headerBuf;

// Mutex
std::mutex points_mutex;
std::mutex odom_mutex;
std::mutex buf_mutex;
std::mutex graph_mutex;
std::mutex flag_mutex;

// median pointcloud
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFull;
pcl::PointCloud<pcl::PointXYZI>::Ptr final;
pcl::PointCloud<pcl::PointXYZI>::Ptr tmp;
pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterScancontext;

// Scancontext
SCManager scManager;

// gtsam optimization
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::noiseModel::Diagonal::shared_ptr priorNoisemodel;
gtsam::noiseModel::Diagonal::shared_ptr odomNoisemodel;
gtsam::noiseModel::Diagonal::shared_ptr loopNoisemode;
gtsam::noiseModel::Base::shared_ptr  robustNoiseModel;
gtsam::Values initials;
gtsam::Values results;
gtsam::Pose3 lastPose;
gtsam::Pose3 currentPose;

std_msgs::Header header;
bool first_msg = true;
bool optimize_flag = false;
int odom_num = 0;
int laser_num = 0;
int history_id = -1;
bool zero_flag = true;

void init()
{
    // allocate memory
    laserCloudFull.reset(new pcl::PointCloud<pcl::PointXYZI>);
    final.reset(new pcl::PointCloud<pcl::PointXYZI>);
    tmp.reset(new pcl::PointCloud<pcl::PointXYZI>);

    // set voxel filter
    float filter_size = 0.4;
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);

    // noise model
    //gtsam::Vector6 PriorVector;
    gtsam::Vector PriorVector(6);
    PriorVector << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6; 
    priorNoisemodel = gtsam::noiseModel::Diagonal::Variances(PriorVector);
    gtsam::Vector6 Constraint(6);
    Constraint << 0.10, 0.10, 0.10, 0.10, 0.10, 0.10;
    odomNoisemodel = gtsam::noiseModel::Diagonal::Variances(Constraint);
    loopNoisemode = gtsam::noiseModel::Diagonal::Variances(Constraint);
    robustNoiseModel = gtsam::noiseModel::Robust::Create(
                            gtsam::noiseModel::mEstimator::Cauchy::Create(1),
                            loopNoisemode);
}

gtsam::Pose3 msg2Pose(nav_msgs::OdometryConstPtr &msg)
{
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    Eigen::Vector3d t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    
    gtsam::Rot3 rot(q.w(), q.x(), q.y(), q.z());
    gtsam::Point3 trans(t.x(), t.y(), t.z());
    return gtsam::Pose3(rot, trans);
}

gtsam::Pose3 eigen2Pose(const Eigen::Matrix4f& transform)
{
    Eigen::Matrix3d rotation;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            rotation(i,j) = transform(i,j);

    Eigen::Quaterniond q(rotation);
    Eigen::Vector3d t(transform(0,3), transform(1,3), transform(2,3));
   
    gtsam::Rot3 rot(q.w(), q.x(), q.y(), q.z());
    gtsam::Point3 trans(t.x(), t.y(), t.z());
    return gtsam::Pose3(rot, trans);
}

void GraphOptimization()
{
    ros::Rate loop(1);
    while (ros::ok())
    {
        if(optimize_flag == false)
            continue;
        
        ROS_INFO("Start optimizing...");
        results.clear();

        graph_mutex.lock();
        
        gtsam::LevenbergMarquardtParams param;
        param.maxIterations = 60;
        param.setVerbosity("TERMINATION");  // this will show info about stopping conditions
        param.setVerbosity("SILENT");
        gtsam::LevenbergMarquardtOptimizer optimizer(gtSAMgraph, initials, param);
        results = optimizer.optimize();
 
        nav_msgs::Path path;
        int id = 0;
        for(const gtsam::Values::ConstKeyValuePair& key_value: results)
        {
            gtsam::Pose3 transform = key_value.value.cast<gtsam::Pose3>();
            Eigen::Quaterniond q = transform.rotation().toQuaternion();
            Eigen::Vector3d t = transform.translation();        

            initials.update(id, transform);
            
            geometry_msgs::PoseStamped posestamp;
            posestamp.header = headerBuf[id];
            posestamp.pose.orientation.w = q.w();
            posestamp.pose.orientation.x = q.x();
            posestamp.pose.orientation.y = q.y();
            posestamp.pose.orientation.z = q.z();
            posestamp.pose.position.x = t.x();
            posestamp.pose.position.y = t.y();
            posestamp.pose.position.z = t.z();

            path.poses.push_back(posestamp);

            id++;
        }

        path.header = headerBuf[id-1];
        pub_path.publish(path);

        graph_mutex.unlock();

        flag_mutex.lock();
        optimize_flag = false;
        flag_mutex.unlock();

        loop.sleep();
    }
}


bool geometryVerfication(int curr_id, int match_id, gtsam::Pose3& relative)
{
    double residual_q = 0.0;
    double residual_t = 0.0;

    gtsam::Pose3 pose1 = initials.at<gtsam::Pose3>(curr_id);
    gtsam::Pose3 pose2 = initials.at<gtsam::Pose3>(match_id);
    gtsam::Pose3 transform1 = pose2.between(pose1);
    gtsam::Pose3 transform = transform1.between(relative);
    residual_t += std::abs(transform.translation().x()) + std::abs(transform.translation().y()) + 
                        std::abs(transform.translation().z());
    residual_q += std::abs(transform.rotation().toQuaternion().w()-1) + std::abs(transform.rotation().toQuaternion().x()) 
                    + std::abs(transform.rotation().toQuaternion().y()) + std::abs(transform.rotation().toQuaternion().z());
    
    if(residual_q > 0.02 || residual_t > 0.5)
        return false;
    
    return true;
}

void odom_callback(const nav_msgs::OdometryConstPtr& msg)
{
    odom_mutex.lock();
    odometryBuf.push(msg);
    odom_num++;
    odom_mutex.unlock();
}

void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    points_mutex.lock();
    fullPointsBuf.push(msg);
    laser_num++;
    points_mutex.unlock();
}

void detect_loop_closure()
{
    if(fullPointsBuf.empty() || odometryBuf.empty())
        return;

    buf_mutex.lock();
    
    double time1 = fullPointsBuf.front()->header.stamp.toSec();
    double time2 = odometryBuf.front()->header.stamp.toSec();
    if(!fullPointsBuf.empty() && std::abs(time1 - time2) > 0.005)
    {
        fullPointsBuf.pop();
        buf_mutex.unlock();
        return;
    }

    if(!odometryBuf.empty() && std::abs(time1 - time2) > 0.005)
    {
        odometryBuf.pop();
        buf_mutex.unlock();
        return;
    }
    //std::cout << "laser: " << laser_num << "\nodom: " << odom_num << std::endl; 

    buf_mutex.unlock();

// Process pointcloud msg
    tmp->clear();
    final->clear();
    laserCloudFull->clear();
    
    points_mutex.lock();
    pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFull);
    fullPointsBuf.pop();
    points_mutex.unlock();
    
    // DownSample Pointcloud
    downSizeFilterScancontext.setInputCloud(laserCloudFull);
    downSizeFilterScancontext.filter(*tmp);
    laserCloudBuf.push_back(tmp);
    
    // Make Scancontext
    scManager.makeAndSaveScancontextAndKeys(*tmp);
// end process pointcloud
    
// Process odometry msg 
    odom_mutex.lock();
    currentPose = msg2Pose(odometryBuf.front()); 
    header = odometryBuf.front()->header;
    odometryBuf.pop();
    odom_mutex.unlock();
    headerBuf.push_back(header);
// end process odometry

    graph_mutex.lock();
    int key = laserCloudBuf.size()-1;
    initials.insert(key, currentPose);

    if(first_msg)
    {   
        // add PriorFactor; to graph
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(key, currentPose, priorNoisemodel));
        first_msg = false;
    }
    else
    {
        //gtsam::Pose3 increment =  lastPose.inverse() * currentPose;
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(key-1, key, lastPose.between(currentPose), odomNoisemodel));
    }
    graph_mutex.unlock();

    int current_id = laserCloudBuf.size()-1;

    // Detect loop closure
    auto result = scManager.detectLoopClosureID();
    history_id = result.first;

    // Verify loop closure
    if(history_id != -1)
    {
        if(!zero_flag)
            return;
        if(history_id == 0)
            zero_flag = false;

        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(20);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        icp.setInputSource(tmp);
        icp.setInputTarget(laserCloudBuf[history_id]);
        icp.align(*final);

        if(icp.hasConverged() == true && icp.getFitnessScore() < 0.001)
        {
            ROS_INFO("[Loop info]: from %d to %d", current_id, history_id);

            Eigen::Matrix4f loop = icp.getFinalTransformation();
            gtsam::Pose3 poseFrom = eigen2Pose(loop);
            gtsam::Pose3 poseTo = gtsam::Pose3::identity();

            //if(!geometryVerfication(current_id, history_id, poseFrom))
            //    return;
            graph_mutex.unlock();
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(current_id, history_id, poseFrom.between(poseTo), loopNoisemode));
            graph_mutex.unlock();

            flag_mutex.lock();
            optimize_flag = true;
            flag_mutex.unlock();
        }
    }
    
    lastPose = currentPose;
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "sc_optimization");
    ros::NodeHandle nh("~");
 
    ros::Subscriber sub_odom = nh.subscribe<nav_msgs::Odometry>("/odom", 1000, odom_callback);
    ros::Subscriber sub_pointcloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1000, pointcloud_callback);
    
    pub_path = nh.advertise<nav_msgs::Path>("final_path", 10);

    init();

    std::thread optimization_thread{GraphOptimization};

    ros::Rate rate(30);
    while (ros::ok())
    {
        detect_loop_closure();
        ros::spinOnce();
        rate.sleep();
    }
    
    optimization_thread.join();

    return 0;
}
