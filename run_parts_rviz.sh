#!/usr/bin/env bash
# 우리 방식(규칙 + DLS IK)으로 계획한 5개 부품을 RViz 창 5개로 동시에 재생한다 (반복 재생).
# 창마다 ROS 도메인을 따로 써서 서로 섞이지 않는다 (111~115). 창을 모두 닫거나 Ctrl+C 로 종료.
set -eo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash; source /home/jhin/doosan_ws/install/setup.bash; source /home/jhin/spraygun_ws/install/setup.bash
export ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR=/tmp/gantry_parts_rviz_logs
trap 'jobs -pr | xargs -r kill 2>/dev/null || true' EXIT
# 창 이름(rviz 설정)  도메인  URDF  계획 파일  경로 표시 인자
PARTS=(
  "1_bumper|111|paint_cell_231.urdf|planner/out_gun_off_231_speed/trajectory_bumper.npz|bumper --csv $DIR/ES300h_front_bumper/painting_trajectory_adjusted.csv --urdf $DIR/planner/paint_cell_collision_231.urdf --no-gantry-height"
  "2_left_fender|112|paint_cell_fender.urdf|planner/out_part_fender_speed/trajectory_fender.npz|fender --urdf $DIR/planner/paint_cell_collision_fender.urdf --no-gantry-height"
  "3_right_fender|113|paint_cell_rfender.urdf|planner/out_part_rfender_speed/trajectory_rfender.npz|rfender --urdf $DIR/planner/paint_cell_collision_rfender.urdf --no-gantry-height"
  "4_hood|114|paint_cell_hood.urdf|planner/out_part_hood_speed/trajectory_hood.npz|hood --urdf $DIR/planner/paint_cell_collision_hood.urdf --no-gantry-height"
  "5_trunk|115|paint_cell_trunk.urdf|planner/out_part_trunk_speed/trajectory_trunk.npz|trunk --csv $DIR/ES300h_trunk/painting_trajectory_reordered.csv --urdf $DIR/planner/paint_cell_collision_trunk.urdf --no-gantry-height"
)
# 인자로 고를 수 있다: ./run_parts_rviz.sh bumper  (bumper, left_fender, right_fender, hood, trunk 중 원하는 것만, 없으면 5개 전부)
for spec in "${PARTS[@]}"; do
  IFS='|' read -r name dom urdf plan path <<< "$spec"
  if [[ $# -gt 0 ]]; then pick=0; for want in "$@"; do [[ "${name#*_}" == "$want" ]] && pick=1; done; [[ $pick == 1 ]] || continue; fi
  if [[ ! -f "$DIR/$plan" ]]; then echo "건너뜀: $name (계획 파일 없음: $plan)"; continue; fi
  params="/tmp/gantry_parts_${name}.yaml"
  python3 - "$DIR/$urdf" "$params" <<'PY'
import sys, yaml
yaml.safe_dump({'robot_state_publisher': {'ros__parameters': {'robot_description': open(sys.argv[1], encoding='utf-8').read()}}}, open(sys.argv[2], 'w', encoding='utf-8'))
PY
  (
    export ROS_DOMAIN_ID=$dom
    ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file "$params" >/dev/null 2>&1 &
    python3 "$DIR/planner/play_trajectory.py" --file "$DIR/$plan" >/dev/null 2>&1 &
    python3 "$DIR/publish_path.py" $path --traj "$DIR/$plan" >/dev/null 2>&1 &
    rviz2 -d "$DIR/rviz_parts/$name.rviz" >/dev/null 2>&1 || echo "RViz 비정상 종료: $name (코드 $?)"
    # 창을 닫으면 이 창의 보조 프로세스를 모두 끈다 (ros2 run 이 띄운 실제 노드까지)
    kill $(jobs -p) 2>/dev/null || true
    pkill -KILL -f "params-file $params" 2>/dev/null || true
    pkill -KILL -f "play_trajectory.py --file $DIR/$plan" 2>/dev/null || true
    pkill -KILL -f "publish_path.py $path" 2>/dev/null || true
  ) &
  echo "시작: $name (ROS_DOMAIN_ID=$dom)"
  sleep 1
done
wait
