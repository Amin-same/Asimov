import time
import mujoco
import mujoco.viewer

mymodel = mujoco.MjModel.from_xml_path('pendulum.xml')
modelData = mujoco.MjData(mymodel)

cam = mujoco.MjvCamera()                         # mjvCamera
opt = mujoco.MjvOption()                         # mjvOption
scn = mujoco.MjvScene(mymodel, maxgeom=1000)       # mjvScene

arr_view = [-90, -16.6, 4.8, 0.01837, 0.0011, 1.2422]

# spec = mujoco.MjSpec.from_file('pendulum.xml')  # nötig, nur wenn man dynamisch und während Simulation was ändern will

# Aktuator über seinen XML-Namen bestimmen

actNum = mymodel.actuator_actnum.shape[0]
print("Number of actuators in Model: " + str(actNum))
print(mymodel.actuator(0).name)
print(mymodel.actuator(1).name)
print(mymodel.actuator(2).name)
targetPos = 3.14


def controller(myModel, modelData):
  #ctrl[0] = torque actuator
  modelData.ctrl[0] = -100*(modelData.qpos[0]-0)-10*modelData.qvel[0]  # PD control


modelData.qpos[0] = -3.14/2
mujoco.set_mjcb_control(controller)

with mujoco.viewer.launch_passive(mymodel, modelData, show_left_ui=False, show_right_ui=True) as viewer:
  viewer.cam.azimuth = arr_view[0]
  viewer.cam.elevation = arr_view[1]
  viewer.cam.distance = arr_view[2]
  viewer.cam.lookat[:] = arr_view[3:]
  #print(viewer.cam)
  # Close the viewer automatically after 40 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 40:
    step_start = time.time()
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(mymodel, modelData)
    # current camera position:
    # print(viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance, viewer.cam.lookat[0], viewer.cam.lookat[1], viewer.cam.lookat[2])
    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(modelData.time % 2)
    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()
    # Rudimentary time keeping, will drift relative to wall clock.
    #time_until_next_step = myModel.opt.timestep - (time.time() - step_start)
    #if time_until_next_step > 0:
     # time.sleep(time_until_next_step)
    time.sleep(1/100)

