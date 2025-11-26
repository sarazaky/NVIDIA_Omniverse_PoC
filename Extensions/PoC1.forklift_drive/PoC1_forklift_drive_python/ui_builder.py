# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import omni.timeline
import omni.ui as ui
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaacsim.gui.components.ui_utils import get_style



#Custom
from pxr import UsdGeom, Usd
import numpy as np, math
import omni.usd
import omni.kit.commands
import scipy.spatial.transform
import time
import omni.kit.commands

FORKLIFT_PATH = "/World/forklift_b"
FORKLIFT_LIFT_PATH = "/World/forklift_b/lift"
SWIVEL_JOINT = "back_wheel_swivel"
DRIVE_JOINT = "back_wheel_drive"
LIFT_JOINT = "lift_joint"
DROP_PALLET = '/World/Pallet_Drop'
FORWARD_SPEED = 10
ROTATION_SPEED = 2
STOP_DISTANCE = 0.005
START_DISTANCE = 0.05
STOP_ANGLE = 0.000005
DECEL_DISTANCE = 1
ACCEL = 0.1





class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []

        # UI elements created using a UIElementWrapper from isaacsim.gui.components.element_wrappers
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()
        # self._timeline_sub = self._timeline.add_timeline_event_listener(self.on_timeline_event)

        # Run initialization for the provided example
        self._on_init()

        #Custom
        print('init')



    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass
        

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        pass
        # forklift = SingleArticulation(FORKLIFT_PATH)
        # forklift_pos, forklift_orientation = forklift.get_world_pose()
        # print(f"[{forklift_pos[0]}, {forklift_pos[1]}],")

    def add_rotation_plan(self, target_pos, forklift_pos, forklift_forward_axis, forklift_forward_sign):
        # Convert quaternion → yaw (theta_current)
        if forklift_forward_sign==1 and forklift_forward_axis==0:
            theta_current = np.radians(0)
        elif forklift_forward_sign==-1 and forklift_forward_axis==0:
            theta_current = np.radians(180)
        elif forklift_forward_axis==1:
            theta_current = np.radians(9 * forklift_forward_sign)

        n_forklift_forward_axis = 1-forklift_forward_axis
        n_forward_distance = target_pos[n_forklift_forward_axis] - forklift_pos[n_forklift_forward_axis]
        distance_sign = np.sign(n_forward_distance)


        if distance_sign==1 and n_forklift_forward_axis==0:
            theta_target = np.radians(0)
        elif distance_sign==-1 and n_forklift_forward_axis==0:
            theta_target = np.radians(180)
        elif n_forklift_forward_axis==1:
            theta_target = np.radians(90 * distance_sign)

        # Shortest angle difference
        theta_diff = (theta_target - theta_current + math.pi) % (2 * math.pi) - math.pi
        req_sign = 1 if theta_diff > 0 else -1



        # Add forward move, to mitigate location change after rotation
        req_distance = 0.99 if req_sign==1 else 0.81
        req_distance =req_distance  * forklift_forward_sign

        if (len(self.forklift_movement_plan)>0) and (self.forklift_movement_plan[-1]['fun_name'] == self.move_forward):
            self.forklift_movement_plan[-1]['meta_data']['target_pos'][forklift_forward_axis] = self.forklift_movement_plan[-1]['meta_data']['target_pos'][forklift_forward_axis]+req_distance
        else:
            current_target_position = target_pos.copy()
            current_target_position[forklift_forward_axis] = current_target_position[forklift_forward_axis]+req_distance
            self.forklift_movement_plan.append({
                'fun_name': self.move_forward,
                'meta_data': {
                    'target_pos': current_target_position.copy(),
                    'movement_axis': forklift_forward_axis,
                    'forklift_forward_sign': forklift_forward_sign
                }
            })
        

        
       


        self.forklift_movement_plan.append({
            'fun_name': self.rotate,
            'meta_data': {
                'req_sign': req_sign,
                'target_axis': n_forklift_forward_axis
            }
        })
        forklift_forward_sign = (forklift_forward_axis - (1-forklift_forward_axis) ) * req_sign * forklift_forward_sign

        forklift_forward_axis = 1- forklift_forward_axis

        return forklift_forward_axis, forklift_forward_sign
    



    def build_movement_plan(self, forklift, target_path):
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # get target_prim position
        target_prim = get_prim_at_path(target_path)
        target_world_transform = xform_cache.GetLocalToWorldTransform(target_prim)
        target_pos = target_world_transform.ExtractTranslation()
        target_pos = np.array([target_pos[0], target_pos[1], target_pos[2]])
        # Get forklift live pose
        forklift_pos, forklift_orientation = forklift.get_world_pose()

        fork_insert_sign = np.sign((target_pos[1]-forklift_pos[1])) #Temp(always in y), leave space for the fork

        target_pos[1] = target_pos[1] - 2.8 * fork_insert_sign #Temp(always in y), leave space for the fork

        forklift_forward_axis, forklift_forward_sign, _ = self.get_forklift_forward_direction(forklift_orientation)
        forward_distance = target_pos[forklift_forward_axis]-forklift_pos[forklift_forward_axis]
        forward_distance_abs = abs(forward_distance)

        n_forklift_forward_axis = 1- forklift_forward_axis
        n_forward_distance = target_pos[n_forklift_forward_axis]-forklift_pos[n_forklift_forward_axis]
        n_forward_distance_abs = abs(n_forward_distance)

        if forward_distance_abs>START_DISTANCE:
            self.forklift_movement_plan.append({
                'fun_name': self.move_forward,
                'meta_data': {
                    'target_pos': target_pos.copy(),
                    'movement_axis': forklift_forward_axis,
                    'forklift_forward_sign': forklift_forward_sign
                }
            })

        
        if n_forward_distance_abs>START_DISTANCE:
            forklift_forward_axis, forklift_forward_sign = self.add_rotation_plan(target_pos, forklift_pos, forklift_forward_axis, forklift_forward_sign)
            
            self.forklift_movement_plan.append({
                'fun_name': self.move_forward,
                'meta_data': {
                    'target_pos': target_pos.copy(),
                    'movement_axis': forklift_forward_axis,
                    'forklift_forward_sign': forklift_forward_sign
                }
            })

        if forklift_forward_axis != 1 : # TEMP, assume pallets are always in y, and the forklift is either in x, -x, y, and not in -y
            # Add forward move, to mitigate location change after rotation
            forklift_forward_axis, forklift_forward_sign = self.add_rotation_plan(target_pos, forklift_pos, forklift_forward_axis, forklift_forward_sign)
            
        # INSERT FORK
        current_target_pos = target_pos.copy()
        current_target_pos[forklift_forward_axis] = current_target_pos[forklift_forward_axis] + 0.9 * fork_insert_sign
        self.forklift_movement_plan.append({
                    'fun_name': self.move_forward,
                    'meta_data': {
                        'target_pos': current_target_pos,
                        'movement_axis': forklift_forward_axis,
                        'forklift_forward_sign': forklift_forward_sign
                    }
                })
        
        # LIFT OR DROP BOX
        lift = target_path != DROP_PALLET
        self.forklift_movement_plan.append({
                    'fun_name': self.move_fork,
                    'meta_data': {
                        'lift': lift
                    }
                })
        if not lift:
            self.forklift_movement_plan.append({
                    'fun_name': self.move_forward,
                    'meta_data': {
                        'target_pos': target_pos,
                        'movement_axis': forklift_forward_axis,
                        'forklift_forward_sign': forklift_forward_sign
                    }
                })
        

        print('FORKLIFT MOVEMENT PLAN')
        print(self.forklift_movement_plan)
            

    def on_physics_step(self, step):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        if self.forklift is None:
            return
        if len(self.forklift_movement_plan)==0: #Create a plan for the next target
            if len(self.target_paths)>0:
                self.build_movement_plan(self.forklift, self.target_paths[0])
                self.target_paths.pop(0)
                if len(self.forklift_movement_plan)==0: # The new plan is empty
                    return
            else:
                return
            
        func = self.forklift_movement_plan[0]['fun_name']     # the function to call
        arg = self.forklift_movement_plan[0]['meta_data']     # the parameter to pass
        func(arg)      
        

    def get_forklift_forward_direction(self, forklift_orientation):
        forklift_orientation = [forklift_orientation[1], forklift_orientation[2], forklift_orientation[3], forklift_orientation[0]]
        r = scipy.spatial.transform.Rotation.from_quat(forklift_orientation)
        local_forward = np.array([1, 0, 0])  # or [0, 1, 0] depending on your forklift model
        world_forward = r.apply(local_forward)
        world_forward[2] = 0  # zero Z if Z is vertical
        world_forward /= np.linalg.norm(world_forward)
        current_forward_axis = np.argmax(np.abs(world_forward))
        
        current_forward_sign = np.sign(world_forward[current_forward_axis])
        return current_forward_axis, current_forward_sign, world_forward



    def move_forward(self, meta_data):
        target_pos = meta_data['target_pos']
        axis = meta_data['movement_axis']
        forklift_sign = meta_data['forklift_forward_sign']

        forklift_pos, forklift_orientation = self.forklift.get_world_pose()

        distance = target_pos[axis] - forklift_pos[axis]
        distance_abs = abs(distance)
        sign = np.sign(distance)
        
        desired_speed = FORWARD_SPEED
        # Slow down when close to target
        if distance_abs < DECEL_DISTANCE:
            desired_speed *= (distance_abs / DECEL_DISTANCE)  # scale down smoothly

        # Acceleration / deceleration smoothing
        if self.current_speed < desired_speed:
            self.current_speed = min(self.current_speed + ACCEL, desired_speed)
        else:
            self.current_speed = max(self.current_speed - ACCEL, desired_speed)
            self.current_speed = max(0.5, self.current_speed )


        if distance_abs < STOP_DISTANCE:
            self.current_speed = 0.0
            self.forklift_movement_plan.pop(0)
            print(forklift_pos)

        angle = 0.0
        speed = self.current_speed * sign * forklift_sign
        swivel_idx = self.forklift.get_dof_index(SWIVEL_JOINT)
        drive_idx = self.forklift.get_dof_index(DRIVE_JOINT)
        action = ArticulationAction(
            joint_positions=np.array([angle, 0.0]),
            joint_velocities=np.array([0.0, speed]),
            joint_indices=np.array([swivel_idx, drive_idx]),
        )
        self.forklift.apply_action(action)


    def rotate(self, meta_data):
        forklift_pos, forklift_orientation = self.forklift.get_world_pose()
        forklift_forward_axis, forklift_forward_sign, world_forward = self.get_forklift_forward_direction(forklift_orientation)

        world_forward_d = world_forward[meta_data['target_axis']]
        if (1-abs(world_forward_d)) < STOP_ANGLE:
            speed = 0
            self.forklift_movement_plan.pop(0)
            print(forklift_pos, ROTATION_SPEED)
        else:
            speed = ROTATION_SPEED * min(1, np.power((1-abs(world_forward_d)), 0.2))
            speed = ROTATION_SPEED
            speed = speed * meta_data['req_sign']
        # print('speed', speed)
        swivel_idx = self.forklift.get_dof_index(SWIVEL_JOINT)
        drive_idx = self.forklift.get_dof_index(DRIVE_JOINT)
        action = ArticulationAction(
            joint_positions=np.array([np.radians(90), 0.0]),
            joint_velocities=np.array([0.0, speed]),
            joint_indices=np.array([swivel_idx, drive_idx]),
        )
        self.forklift.apply_action(action)
        # print(f'[{forklift_pos[0]}, {forklift_pos[1]}],')

    def move_fork(self, meta_data):
        if meta_data['lift']:
            tar = 0.29
        else:
            tar = 0.2

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # get lift position
        lift_prim = get_prim_at_path(FORKLIFT_LIFT_PATH)
        target_world_transform = xform_cache.GetLocalToWorldTransform(lift_prim)
        lift_pos = target_world_transform.ExtractTranslation()
        lift_pos = np.array([lift_pos[0], lift_pos[1], lift_pos[2]])

        

        if lift_pos[2]<tar-0.05:
            self.lift_current_speed += 0.05
        elif lift_pos[2]>tar+0.05:
            self.lift_current_speed -= 0.05
        else:
            self.lift_current_speed = self.lift_current_speed
            self.forklift_movement_plan.pop(0)
            print('DONE YAY')

        print(lift_pos, self.lift_current_speed)
        lift_idx = self.forklift.get_dof_index(LIFT_JOINT)
        action = ArticulationAction(
            joint_positions=np.array([0.0]),
            joint_velocities=np.array([self.lift_current_speed]),
            joint_indices=np.array([lift_idx]),
        )
        self.forklift.apply_action(action)
        


    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(omni.usd.StageEventType.ASSETS_LOADED):  # Any asset added or removed
            pass
        elif event.type == int(omni.usd.StageEventType.SIMULATION_START_PLAY):  # Timeline played
            # Treat a playing timeline as a trigger for selecting an Articulation
            if self.forklift is None:
                self.forklift = SingleArticulation(FORKLIFT_PATH)
                self.forklift.initialize()
            self._physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_physics_step)
            print('will play')
        elif event.type == int(omni.usd.StageEventType.SIMULATION_STOP_PLAY):  # Timeline stopped
            # Ignore pause events
            if self._timeline.is_stopped():
                print('WILL STOP')
                if self._physics_sub:
                    self._physics_sub.unsubscribe()
                    self._physics_sub = None
                self._on_init()
        

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from isaacsim.gui.components.element_wrappers implement a cleanup function that should be called
        """
        print('Will Clean')
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()
        
        if self._physics_sub:
            self._physics_sub.unsubscribe()
            self._physics_sub = None

        if not self._timeline.is_stopped():
            self._timeline.stop()

        self._on_init()

        

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """

        self._select_by_id_frame = CollapsableFrame("Enter Box Id/s", collapsed=False)

        with self._select_by_id_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # Guidance label
                ui.Label("Enter the Prim ID below and click 'Highlight Items' to highlight it, or Start Pickup to initiate the movement.")
        

                # Editable text input
                self._prim_id_input_model = ui.SimpleStringModel("")
                self._prim_id_input = ui.StringField(model=self._prim_id_input_model)
                self._prim_message_model = ui.SimpleStringModel("")

                # Button inside HStack
                with ui.HStack():
                    ui.Button("Highlight Items", clicked_fn=self._on_select_prim_by_id)

                # Button inside HStack
                with ui.HStack():
                    ui.Button("Start Pickup", clicked_fn=self._on_start_pickup)

                print('Done')
                # Feedback label
                self.feedback_label = ui.Label("")




    def get_target_paths(self):
        """
        Read the text field and return a list of prim paths
        for all comma-separated ids that exist in the stage.
        Example input: 'PALLET_1, PALLET_2, BOX_A'
        """
        input_text = self._prim_id_input_model.get_value_as_string().strip()
        if not input_text:
            self.feedback_label.text = "Please enter one or more IDs!!"
            return []

        # IDs in the same order the user typed them
        requested_ids = [s.strip() for s in input_text.split(",") if s.strip()]
        if not requested_ids:
            self.feedback_label.text = "Please enter one or more IDs!!"
            return []

        stage = omni.usd.get_context().get_stage()

        # Build id → prim_path map in a single traverse
        id_to_path = {}
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            attr = prim.GetAttribute("id")
            if attr and attr.HasAuthoredValue():
                val = str(attr.Get())
                if val in requested_ids and val not in id_to_path:
                    id_to_path[val] = prim.GetPath().pathString

        found_paths = []
        missing_ids = []
        for rid in requested_ids:
            if rid in id_to_path:
                found_paths.append(id_to_path[rid])
            else:
                missing_ids.append(rid)

        if not found_paths:
            self.feedback_label.text = "No prims found for the given IDs."
            return []

        if missing_ids:
            self.feedback_label.text = (
                f"Found {len(found_paths)} item(s). "
                f"Missing IDs: {', '.join(missing_ids)}"
            )
        else:
            self.feedback_label.text = f"Found {len(found_paths)} item(s)."

        return found_paths



    def _on_select_prim_by_id(self):
        paths = self.get_target_paths()
        if not paths:
            return

        # Select all matched prims
        ctx = omni.usd.get_context()
        ctx.get_selection().set_selected_prim_paths(paths, True)

        self.feedback_label.text += f"  Selected {len(paths)} prim(s)."

    def _on_start_pickup(self):
    
        if len(self.target_paths) > 0:
            return

        pallet_paths = self.get_target_paths()
        if not pallet_paths:
            return

        # Start / resume timeline
        if not self._timeline.is_playing():
            self._timeline.play()

        # Build full sequence:
        # PALLET_1 -> DROP_PALLET -> PALLET_2 -> DROP_PALLET -> ...
        sequence = []
        for p in pallet_paths:
            sequence.append(p)          # go pick this pallet
            sequence.append(DROP_PALLET)  # then go drop it

        self.target_paths = sequence
        self.feedback_label.text += (
            f"  Pickup plan for {len(pallet_paths)} pallet(s) created."
        )

        



    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Replaced/Deleted
    ######################################################################################

    def _on_init(self):
        self.articulation = None
        self._physics_sub = None
        self.forklift = None
        self.forklift_movement_plan = []
        self.target_paths = []
        self.current_speed = 0.0
        self.lift_current_speed = 0.0
