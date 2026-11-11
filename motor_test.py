
from pytrinamic.connections import UsbTmclInterface
from pytrinamic.tmcl import TMCLCommand
import random

# Change this to the COM port assigned by Windows
COM_PORT = "COM5"


import time
from pytrinamic.connections import ConnectionManager

# TMCL command opcodes
MST = 3
MVP = 4
SAP = 5
GAP = 6
RFS = 13

target_pos = 0
act_pos = 1
target_spd = 4
max_acc = 5
pos_reach_flg = 8
Ref_Search_Mode = 193
Ref_search_spd = 194


search_speed_deg_per_sec = 5
move_spd_deg_per_sec = 10
mov_acc_deg_per_sec = 5000
microsteps_per_deg = 256*(200/360)

class Motor:
    def __init__(self,COM_PORT,motor_number,steps_per_degree):
        self.COM_PORT = COM_PORT
        self.home_defined = False
        self.motor = UsbTmclInterface(COM_PORT)
        self.motor_number = motor_number
        self.steps_per_degree = steps_per_degree
        self.send_command(SAP,Ref_Search_Mode,1)
        self.send_command(SAP,Ref_search_spd,self.angle_to_steps(search_speed_deg_per_sec))
        self.send_command(RFS, 0,0)
        self.send_command(SAP,target_spd,self.angle_to_steps(move_spd_deg_per_sec))
        self.send_command(SAP,max_acc,self.angle_to_steps(mov_acc_deg_per_sec))

    def send_command(self, opcode, op_type=0, value=0):
        response = self.motor.send(
            opcode=opcode,
            op_type=op_type,
            motor=self.motor_number,
            value=value
        )

        if not response.is_valid:
            raise ValueError
        else:
            return response

    def set_axis_parameter(self, parameter, value):
        self.send_command(self.SAP, parameter, value)


    def get_axis_parameter(self, parameter):
        response = self.send_command(self.GAP, parameter, 0)
        return response.value


    def move_absolute(self, position):
        self.send_command(self.MVP, 0, position)

    def steps_to_angle(self, steps):
        return int(steps/self.steps_per_degree)

    def angle_to_steps(self, angle):
        return int(angle * self.steps_per_degree)

def main():
    motor = Motor(COM_PORT,0,microsteps_per_deg)
    home_sequence_tried = False
    i = 0

    while True:
        try:
            response = motor.send_command(GAP, pos_reach_flg,0)
            
            
            if response.is_valid:
                if response.value == 1:
                    new_pos = random.randint(0,90)
                    response = motor.send_command(MVP, 0, motor.angle_to_steps(new_pos))
            
            if i%20 == 0:
                response = motor.send_command(GAP, act_pos, 0)
                print(motor.steps_to_angle(response.value))    
                   
        finally:
            i = i + 1
    

if __name__ == "__main__":
    main()
