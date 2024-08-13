import tkinter as tk
from tkinter import messagebox
from UserInput import UserInput, UserInputException
from Solve import Solver
from Sokoban import Game


class GUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Sokoban Planning Problem')
        self.frame = tk.Frame(self.window)

        self.frame1 = tk.Frame(self.frame)
        self.frame1.grid(row=0, column=0)

        self.frame2 = tk.Frame(self.frame)
        self.frame2.grid(row=1, column=0)
        
        self.font = 'Helvetica 12'
        self.pady = 3
        self.padx = 2
        self.entry_width = 50
        self.draw_window()

        self.game = None

        #self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.frame.pack()
        self.window.mainloop()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.game:
                self.game.active = False
            self.window.destroy()
        
    def draw_window(self):
        l1 = tk.Label(self.frame1, text='Map Path:', font=self.font)
        l2 = tk.Label(self.frame1, text='Debug Files name:', font=self.font)

        l1.grid(row=0, column=0, sticky='W', pady=self.pady, padx=self.padx)
        l2.grid(row=1, column=0, sticky='W', pady=self.pady, padx=self.padx)

        self.e1 = tk.Entry(self.frame1, font=self.font, width=self.entry_width)
        self.e1.insert(0, 'maps/map4.txt')
        self.e1.grid(row=0, column=1, pady=self.pady, padx=self.padx)

        self.e2 = tk.Entry(self.frame1, font=self.font, width=self.entry_width)
        self.e2.insert(0, 'map4')
        self.e2.grid(row=1, column=1, pady=self.pady, padx=self.padx)       
        
        button = tk.Button(self.frame1, text='Find Plan', font=self.font, command=self.create_theory)
        button.grid(row=2, column=1, pady=self.pady, padx=self.padx)

    def create_theory(self):
        try:
            if self.game:
                self.game.active = False
                for widgets in self.frame2.winfo_children():
                    widgets.destroy()
                    
            path = self.e1.get()
            UserInput.check_file(path)

            name = self.e2.get()
            solution = Solver.find_solution(path, name)

            yes = messagebox.askyesno("Plan","Do you want to visualise the plan?")
            if yes:
                instructions = Solver.parse_answer(solution)
                self.game = Game(self.frame2, path)
                self.game.visualise_plan(instructions)
        except UserInputException as ex:
            messagebox.showerror('User Input Error', str(ex))
        except Exception as ex:
            messagebox.showerror('Unhandled Exception', str(ex))
    

if __name__ == '__main__':
    GUI()
    #Solver.find_solution('maps/map4.txt', 'map4')

