#!/usr/bin/env bash
set -eo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source /home/jhin/doosan_ws/install/setup.bash
source /home/jhin/spraygun_ws/install/setup.bash
MODE="${1:-manual}"
SCENE="$DIR/paint_cell_raised.urdf"
RVIZ_CONFIG="$DIR/paint_cell.rviz"
PLAY_FIRST=0
if [[ "$MODE" == "--approach" ]]; then
  python3 "$DIR/planner/prepare_approach.py"
  SCENE="$DIR/paint_cell_231.urdf"
  PLAN="$DIR/planner/out_approach/trajectory_bumper.npz"
  RVIZ_CONFIG="$DIR/paint_cell_approach.rviz"
  PLAY_FIRST=5
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-105}"
elif [[ "$MODE" == "--first-reference" ]]; then
  python3 "$DIR/planner/first_pose_reference.py"
  SCENE="$DIR/paint_cell_208.urdf"
  PLAN="$DIR/planner/out_first_reference/trajectory_bumper.npz"
  PLAY_FIRST=4
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
elif [[ "$MODE" == "--one-row" ]]; then
  ROW="${2:-6}"
  if [[ "$ROW" != "6" ]]; then echo '낮춘 범퍼의 단일 줄 계산은 현재 6번 줄만 준비되어 있습니다.' >&2; exit 1; fi
  python3 "$DIR/planner/extract_row.py" "$ROW" --source out_row6_208_speed
  SCENE="$DIR/paint_cell_208.urdf"
  PLAN="$DIR/planner/out_single_row/trajectory_bumper.npz"
  PLAY_FIRST=4
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-98}"
elif [[ "$MODE" == "--follow" ]]; then
  SCENE="$DIR/paint_cell_231.urdf"
  PLAN="$DIR/planner/out_gun_off_231_speed/trajectory_bumper.npz"
  [[ -f "$PLAN" ]] || { echo '검토된 계획 파일이 없습니다.' >&2; exit 1; }
  PLAY_FIRST=3
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-97}"
elif [[ "$MODE" == "--first-point" ]]; then
  GUN="${2:-on}"
  SCENE_NAME="paint_cell_raised.urdf"
  if [[ "${3:-}" == "--lowered" ]]; then SCENE_NAME="paint_cell.urdf"; fi
  SCENE="$DIR/$SCENE_NAME"
  if python3 "$DIR/plan_first_point.py" --gun "$GUN" --scene "$SCENE_NAME"; then
    PLAY_FIRST=1
  else
    echo '첫 점 IK 해를 찾지 못해 이전 결과를 재생하지 않고 슬라이더로 표시합니다.' >&2
  fi
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-92}"
else
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-91}"
fi
export ROS_LOCALHOST_ONLY=1
PARAMS="$(mktemp /tmp/gantry_sim_rsp_XXXXXX.yaml)"
cleanup() { jobs -pr | xargs -r kill 2>/dev/null || true; rm -f "$PARAMS"; }
trap cleanup EXIT
python3 - "$SCENE" "$PARAMS" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding='utf-8') as f:
    xml = f.read()
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    yaml.safe_dump({'robot_state_publisher': {'ros__parameters': {'robot_description': xml}}}, f)
PY
ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file "$PARAMS" &
if [[ "$PLAY_FIRST" == 5 ]]; then
  python3 "$DIR/planner/play_trajectory.py" --once --speed 1.0 --start-delay 10 --file "$PLAN" &
elif [[ "$PLAY_FIRST" == 4 ]]; then
  python3 "$DIR/planner/play_trajectory.py" --file "$PLAN" &
elif [[ "$PLAY_FIRST" == 3 ]]; then
  python3 "$DIR/planner/play_trajectory.py" --once --file "$PLAN" &
elif [[ "$PLAY_FIRST" == 2 ]]; then
  python3 "$DIR/play_follow.py" &
elif [[ "$PLAY_FIRST" == 1 ]]; then
  python3 "$DIR/play_first_point.py" &
else
  GUN_LIFT=0.0736
  if [[ "$MODE" == "--first-point" && "$GUN" == "off" ]]; then GUN_LIFT=0.0; fi
  ros2 run joint_state_publisher_gui joint_state_publisher_gui --ros-args \
    -p zeros.gantry_x:=3.781161 -p zeros.gantry_y:=1.862 -p zeros.gun_lift:="$GUN_LIFT" \
    -p zeros.joint_2:=-1.5708 -p zeros.joint_3:=1.5708 &
fi
if [[ "$MODE" == "--approach" ]]; then
  python3 "$DIR/publish_path.py" bumper --csv "$DIR/ES300h_front_bumper/painting_trajectory_adjusted.csv" --row 0 --no-row-label &
elif [[ "$MODE" == "--first-reference" ]]; then
  python3 "$DIR/publish_path.py" bumper --csv "$DIR/ES300h_front_bumper/painting_trajectory_adjusted.csv" --row 6 &
elif [[ "$MODE" == "--one-row" ]]; then
  python3 "$DIR/publish_path.py" bumper --csv "$DIR/ES300h_front_bumper/painting_trajectory_adjusted.csv" --row "$ROW" &
elif [[ "$MODE" == "--follow" ]]; then
  python3 "$DIR/publish_path.py" bumper --csv "$DIR/ES300h_front_bumper/painting_trajectory_adjusted.csv" --traj "$PLAN" &
else
  python3 "$DIR/publish_path.py" bumper &
fi
rviz2 -d "$RVIZ_CONFIG"
