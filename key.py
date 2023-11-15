# Using Keyboard module in Python
#import keyboard
from curses.ascii import alt
from pynput.keyboard import Key, Controller
from pynput.mouse import Button
from pynput.mouse import Controller as mctrl


keyboard = Controller()
mouse = mctrl()

def do_action(current, prev):

    if(current != "Fist"):      
        
        if(current != "Switch Windows" and prev == "Switch Windows"):
            #keyboard.press(Key.alt)
            keyboard.release(Key.alt)
        
        if(current == "Scroll Up"):
            mouse.scroll(0, -2)

        elif(current == "Scroll Down"):
            mouse.scroll(0, 2)

        elif(current == "Backspace"):
            keyboard.press(Key.backspace)

        elif(current == "Enter"):
            if(prev == "Switch Windows"):
                #keyboard.press(Key.alt)
                keyboard.release(Key.alt)

            keyboard.press(Key.enter)

        elif(current == "Select All"):
            keyboard.press(Key.ctrl_l)
            keyboard.press('a')
            keyboard.release('a')
            keyboard.release(Key.ctrl_l)

        elif(current == "Copy"):
            keyboard.press(Key.ctrl_l)
            keyboard.press('c')
            keyboard.release('c')
            keyboard.release(Key.ctrl_l)

        elif(current == "Paste"):
            keyboard.press(Key.ctrl_l)
            keyboard.press('v')
            keyboard.release('v')
            keyboard.release(Key.ctrl_l)

        elif(current == "Switch Windows"):
            if(prev != "Switch Windows"):
                keyboard.press(Key.alt)
                keyboard.press(Key.tab)
                keyboard.release(Key.tab)

            elif(prev == "Switch Windows"):
                keyboard.press(Key.tab)
                keyboard.release(Key.tab)

        elif(current == "Hello"):
            keyboard.type("Hello!")

        elif(current == "Maximize"):
            keyboard.press(Key.cmd)
            keyboard.press(Key.up)
            keyboard.release(Key.up)
            keyboard.release(Key.cmd)

        elif(current == "Minimize"):
            keyboard.press(Key.cmd)
            keyboard.press(Key.down)
            keyboard.release(Key.down)
            keyboard.release(Key.cmd)

        elif(current == "Show Desktop"):
            keyboard.press(Key.cmd)
            keyboard.press('d')
            keyboard.release('d')
            keyboard.release(Key.cmd)


        

    #else:
    #   keyboard.type(current + "\n")