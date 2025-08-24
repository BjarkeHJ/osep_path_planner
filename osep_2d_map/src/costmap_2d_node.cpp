#include "costmap_2d.hpp"

ESDF2dCostMapNode::ESDF2dCostMapNode()
: Node("costmap_2d_node"),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  // Declare and retrieve parameters
  this->declare_parameter("resolution", 1.0);     // Cell size in meters
  this->declare_parameter("free_center_radius", 10.0); // Radius of the free center area
  this->declare_parameter("local_map_size", 200.0);    // Local map size (200 m x 200 m)
  this->declare_parameter("global_map_size", 1500.0);  // Global map size (1500 m x 1500 m)
  this->declare_parameter("frame_id", "base_link");    // Map centered at base_link
  this->declare_parameter("safety_distance", 10.0);  // Central safety distance
  this->declare_parameter("costmap_topic", "/local_costmap/costmap");

  resolution_ = this->get_parameter("resolution").as_double();
  free_center_radius_ = this->get_parameter("free_center_radius").as_double();
  local_map_size_ = this->get_parameter("local_map_size").as_double();
  global_map_size_ = this->get_parameter("global_map_size").as_double();
  frame_id_ = this->get_parameter("frame_id").as_string();
  safety_distance_ = this->get_parameter("safety_distance").as_double();
  costmap_topic_ = this->get_parameter("costmap_topic").as_string();
  safety_distance_min_ = safety_distance_ - 0.2 * safety_distance_;
  safety_distance_max_ = safety_distance_ + 0.2 * safety_distance_;


  // Compute grid dimensions
  local_grid_size_ = static_cast<int>(local_map_size_ / resolution_);
  global_grid_size_ = static_cast<int>(global_map_size_ / resolution_);
  local_half_size_ = local_map_size_ / 2.0;
  global_half_size_ = global_map_size_ / 2.0;

  // Initialize the global map
  global_map_.info.resolution = resolution_;
  global_map_.info.width = global_grid_size_;
  global_map_.info.height = global_grid_size_;
  global_map_.info.origin.position.x = -global_half_size_;
  global_map_.info.origin.position.y = -global_half_size_;
  global_map_.info.origin.position.z = 0.0;
  global_map_.info.origin.orientation.w = 1.0;
  global_map_.data.resize(global_grid_size_ * global_grid_size_, -1); // Initialize as unknown

  // Subscribe to the ESDF point cloud topic
  esdf_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/nvblox_node/static_esdf_pointcloud", 10,
    std::bind(&ESDF2dCostMapNode::esdf_callback, this, std::placeholders::_1)
  );

  // Publisher for the global cost map
  global_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("osep/global_costmap/costmap", 10);

  // Publisher for the local cost map
  local_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(costmap_topic_, 10);

  RCLCPP_INFO(this->get_logger(), "ESDF cost map node initialized.");
}

bool ESDF2dCostMapNode::validate_pointcloud2_fields(const sensor_msgs::msg::PointCloud2& msg) {
    return msg.fields.size() >= 4 &&
           msg.fields[0].name == "x" &&
           msg.fields[1].name == "y" &&
           msg.fields[2].name == "z" &&
           msg.fields[3].name == "intensity";
}

pcl::PointCloud<pcl::PointXYZI>::Ptr ESDF2dCostMapNode::convert_to_pcl_cloud(const sensor_msgs::msg::PointCloud2& msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(msg, *cloud);
    return cloud;
}

std::optional<geometry_msgs::msg::TransformStamped> ESDF2dCostMapNode::get_transform_to_odom() {
    try {
        return tf_buffer_.lookupTransform("odom", frame_id_, rclcpp::Time(0));
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "Could not transform %s to odom: %s", frame_id_.c_str(), ex.what());
        return std::nullopt;
    }
}

void ESDF2dCostMapNode::extract_local_from_global(nav_msgs::msg::OccupancyGrid& local_map) {
    for (int y = 0; y < local_grid_size_; ++y) {
        for (int x = 0; x < local_grid_size_; ++x) {
            int global_x = static_cast<int>((local_map.info.origin.position.x + x * resolution_ - global_map_.info.origin.position.x) / resolution_);
            int global_y = static_cast<int>((local_map.info.origin.position.y + y * resolution_ - global_map_.info.origin.position.y) / resolution_);
            int global_index = global_y * global_grid_size_ + global_x;
            int local_index = y * local_grid_size_ + x;
            if (global_x >= 0 && global_x < global_grid_size_ && global_y >= 0 && global_y < global_grid_size_) {
                local_map.data[local_index] = global_map_.data[global_index];
            }
        }
    }
}

void ESDF2dCostMapNode::overwrite_local_with_esdf(nav_msgs::msg::OccupancyGrid& local_map, const pcl::PointCloud<pcl::PointXYZI>& cloud) {
    for (const auto& point : cloud.points) {
        int grid_x = static_cast<int>((point.x - local_map.info.origin.position.x) / resolution_);
        int grid_y = static_cast<int>((point.y - local_map.info.origin.position.y) / resolution_);

        if (grid_x >= 0 && grid_x < local_grid_size_ && grid_y >= 0 && grid_y < local_grid_size_) {
            float distance = point.intensity; // Distance to nearest obstacle from ESDF
            int cost;

            if (distance < 0.0f) {
                // Inside an obstacle or invalid value
                cost = 100; // Maximum cost for obstacle
            } else if (distance <= safety_distance_min_) {
                // Within minimum safety distance
                cost = 100; // Maximum cost (unsafe region)
            } else if (distance <= safety_distance_max_) {
                // Between minimum and maximum safety distance
                // Linearly interpolate cost from 100 to 0
                cost = static_cast<int>(100.0f * (1.0f - (distance - safety_distance_min_) /
                                                  (safety_distance_max_ - safety_distance_min_)));
            } else {
                // Beyond maximum safety distance
                cost = 0; // No cost (safe region)
            }

            // Ensure cost stays within the range [0, 100]
            cost = std::clamp(cost, 0, 100);

            local_map.data[grid_y * local_grid_size_ + grid_x] = cost;
        }
    }
}

nav_msgs::msg::OccupancyGrid ESDF2dCostMapNode::create_local_map(const geometry_msgs::msg::TransformStamped& transform) {
    nav_msgs::msg::OccupancyGrid local_map;
    local_map.info.resolution = resolution_;
    local_map.info.width = local_grid_size_;
    local_map.info.height = local_grid_size_;
    local_map.info.origin.position.x = std::floor((transform.transform.translation.x - local_half_size_ - global_map_.info.origin.position.x) / resolution_) * resolution_ + global_map_.info.origin.position.x;
    local_map.info.origin.position.y = std::floor((transform.transform.translation.y - local_half_size_ - global_map_.info.origin.position.y) / resolution_) * resolution_ + global_map_.info.origin.position.y;
    local_map.info.origin.position.z = 0.0;
    local_map.info.origin.orientation.w = 1.0;
    local_map.data.resize(local_grid_size_ * local_grid_size_, -1); // Initialize as unknown
    return local_map;
}

void ESDF2dCostMapNode::clear_local_center(nav_msgs::msg::OccupancyGrid& local_map) {
    double clear_radius = free_center_radius_; // Radius in meters
    int clear_radius_cells = static_cast<int>(clear_radius / resolution_);

    // Calculate the center of the global map in the local map's coordinate system
    int global_center_x = static_cast<int>((global_map_.info.origin.position.x + global_half_size_ - local_map.info.origin.position.x) / resolution_);
    int global_center_y = static_cast<int>((global_map_.info.origin.position.y + global_half_size_ - local_map.info.origin.position.y) / resolution_);

    for (int y = 0; y < local_grid_size_; ++y) {
        for (int x = 0; x < local_grid_size_; ++x) {
            int dx = x - global_center_x;
            int dy = y - global_center_y;
            if (dx * dx + dy * dy <= clear_radius_cells * clear_radius_cells) {
                local_map.data[y * local_grid_size_ + x] = 0; // Clear cell
            }
        }
    }
}

void ESDF2dCostMapNode::merge_local_to_global(const nav_msgs::msg::OccupancyGrid& local_map) {
    for (int y = 0; y < local_grid_size_; ++y) {
        for (int x = 0; x < local_grid_size_; ++x) {
            int local_index = y * local_grid_size_ + x;
            int global_x = static_cast<int>((local_map.info.origin.position.x + x * resolution_ - global_map_.info.origin.position.x) / resolution_);
            int global_y = static_cast<int>((local_map.info.origin.position.y + y * resolution_ - global_map_.info.origin.position.y) / resolution_);
            int global_index = global_y * global_grid_size_ + global_x;

            if (global_x >= 0 && global_x < global_grid_size_ && global_y >= 0 && global_y < global_grid_size_) {
                if (local_map.data[local_index] != -1) { // If the local cell is observed
                    global_map_.data[global_index] = local_map.data[local_index];
                }
            }
        }
    }
}

void ESDF2dCostMapNode::publish_maps(const nav_msgs::msg::OccupancyGrid& local_map) {
    nav_msgs::msg::OccupancyGrid local_map_copy = local_map;
    local_map_copy.header.stamp = this->now();
    local_map_copy.header.frame_id = "odom";
    local_map_pub_->publish(local_map_copy);

    global_map_.header.stamp = this->now();
    global_map_.header.frame_id = "odom";
    global_map_pub_->publish(global_map_);
}

void ESDF2dCostMapNode::esdf_callback(const sensor_msgs::msg::PointCloud2::SharedPtr esdf_msg)
{
  // Check for required fields before converting
  if (!validate_pointcloud2_fields(*esdf_msg)) {
    RCLCPP_WARN(this->get_logger(), "PointCloud2 does not have expected fields!");
    return;
  }

  auto cloud = convert_to_pcl_cloud(*esdf_msg);

  auto transform = get_transform_to_odom();
  if (!transform) return;

  nav_msgs::msg::OccupancyGrid local_map = create_local_map(*transform);
  extract_local_from_global(local_map);
  overwrite_local_with_esdf(local_map, *cloud);
  clear_local_center(local_map);
  merge_local_to_global(local_map);

  publish_maps(local_map);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ESDF2dCostMapNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
