import tkinter as tk
from tkinter import filedialog as fd
from Renderer import *

class GUI:
    class SaveResetPanel:
        mesh = None
        
        def __init__(self, root, b_height, b_width, f_size):
            panel = tk.Frame(root)
            panel.pack()

            pixelVirtual = tk.PhotoImage(width=1, height=1)

            
            load = tk.Button(panel, text="Load", height=b_height, width=b_width,
                             font=("", f_size, ""), command=GUI.SaveResetPanel.load_mesh)
            load.grid(row=0, column=0, padx=10, pady=10)
      

            reset = tk.Button(panel, text="Reset", height=b_height, width=b_width,
                             font=("", f_size, ""), command=GUI.SaveResetPanel.reset_options)
            reset.grid(row=0, column=1, padx=10, pady=10)

        def load_mesh():
            filename = tk.filedialog.askopenfile(filetypes=(("Wavefront file", "*.obj"),))
            if filename:
                mesh = Mesh(filename.name)
                GUI.renderer.clear()
                GUI.SaveResetPanel.set_mesh(mesh)   
                GUI.SaveResetPanel.reset_options()

        def reset_options():
            mesh = GUI.SaveResetPanel.get_mesh()
            if mesh:
                GUI.TranslationPanel.reset_variables()
                GUI.RotationPanel.reset_variables()
                GUI.ScalePanel.reset_variables()
                GUI.PointLightTransformationPanel.reset_variables()
                GUI.RGBPanel.reset_variables()
                GUI.ReflectionConstantsPanel.reset_variables()
                GUI.Utility_Panel.reset_variables()
                GUI.renderer.clear()
                GUI.renderer.reset_affine_transformations()
                GUI.renderer.render(mesh)
                GUI.renderer.canvas.update()

        def get_mesh():
            return GUI.SaveResetPanel.mesh

        def set_mesh(mesh):
            GUI.SaveResetPanel.mesh = mesh


    class TranslationPanel:
        dx_var = None
        dy_var = None
        dz_var = None
        
        def __init__(self, root, s_width, b_height, b_width, f_size):
            panel = tk.Frame(root)
            panel.pack()

            GUI.TranslationPanel.dx_var = tk.StringVar(root)
            GUI.TranslationPanel.dx_var.set("0.00")

            GUI.TranslationPanel.dy_var = tk.StringVar(root)
            GUI.TranslationPanel.dy_var.set("0.00")

            GUI.TranslationPanel.dz_var = tk.StringVar(root)
            GUI.TranslationPanel.dz_var.set("0.00")

            lb_x = tk.Label(panel, font=("", f_size, ""), text="x-axis")
            lb_x.grid(row=0, column=0, padx=0, pady=5)
            lb_y = tk.Label(panel, font=("", f_size, ""), text="y-axis")
            lb_y.grid(row=0, column=1, padx=0, pady=5)
            lb_z = tk.Label(panel, font=("", f_size, ""), text="z-axis")
            lb_z.grid(row=0, column=2, padx=0, pady=5)

            box_x = tk.Spinbox(panel, from_=-3, to=3, increment=0.01, textvariable=GUI.TranslationPanel.dx_var,
                               width=s_width, font=("", f_size, ""))
            box_x.grid(row=1, column=0)
            
            box_y = tk.Spinbox(panel, from_=-3, to=3, increment=0.01, textvariable=GUI.TranslationPanel.dy_var,
                               width=s_width, font=("", f_size, ""))
            box_y.grid(row=1, column=1)

            box_z = tk.Spinbox(panel, from_=-3, to=3, increment=0.01, textvariable=GUI.TranslationPanel.dz_var,
                               width=s_width, font=("", f_size, ""))
            box_z.grid(row=1, column=2)
            
            translate = tk.Button(panel, text="Translate", height=b_height, width=b_width,
                             font=("", f_size, ""), command=GUI.TranslationPanel.translate_mesh)
            translate.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        def translate_mesh():
            mesh = GUI.SaveResetPanel.get_mesh()
            if mesh:  
                try:
                    dx, dy, dz = float(GUI.TranslationPanel.dx_var.get()), \
                                 float(GUI.TranslationPanel.dy_var.get()), \
                                 float(GUI.TranslationPanel.dz_var.get())
                    GUI.renderer.clear()
                    GUI.renderer.set_translation_matrix(dx, dy, dz)
                    GUI.renderer.render(mesh)
                    GUI.renderer.canvas.update()
                except ValueError:
                    print("Enter a numeric value!")
                    
        def reset_variables():
            GUI.TranslationPanel.dx_var.set("0.00")
            GUI.TranslationPanel.dy_var.set("0.00")
            GUI.TranslationPanel.dz_var.set("0.00")

    class RotationPanel:
        x_deg_var = None
        y_deg_var = None
        z_deg_var = None
        
        def __init__(self, root, s_width, b_height, b_width, f_size):
            panel = tk.Frame(root)
            panel.pack()

            GUI.RotationPanel.x_deg_var = tk.StringVar(root)
            GUI.RotationPanel.x_deg_var.set("0")

            GUI.RotationPanel.y_deg_var = tk.StringVar(root)
            GUI.RotationPanel.y_deg_var.set("0")

            GUI.RotationPanel.z_deg_var = tk.StringVar(root)
            GUI.RotationPanel.z_deg_var.set("0")
            
            lb_x = tk.Label(panel, font=("", f_size, ""), text="x-axis")
            lb_x.grid(row=0, column=0, padx=0, pady=5)
            lb_y = tk.Label(panel, font=("", f_size, ""), text="y-axis")
            lb_y.grid(row=0, column=1, padx=0, pady=5)
            lb_z = tk.Label(panel, font=("", f_size, ""), text="z-axis")
            lb_z.grid(row=0, column=2, padx=0, pady=5)

            box_x = tk.Spinbox(panel, from_=-180, to=180, increment=1,
                               textvariable=GUI.RotationPanel.x_deg_var,
                               width=s_width, font=("", f_size, ""))
            box_x.grid(row=1, column=0)
            
            box_y = tk.Spinbox(panel, from_=-180, to=180, increment=1,
                               textvariable=GUI.RotationPanel.y_deg_var,
                               width=s_width, font=("", f_size, ""))
            box_y.grid(row=1, column=1)

            box_z = tk.Spinbox(panel, from_=-180, to=180, increment=1,
                               textvariable=GUI.RotationPanel.z_deg_var,
                               width=s_width, font=("", f_size, ""))
            box_z.grid(row=1, column=2)
            
            rotatate = tk.Button(panel, text="Rotatate", height=b_height, width=b_width,
                             font=("", f_size, ""), command=GUI.RotationPanel.rotate_mesh)
            rotatate.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        def reset_variables():
            GUI.RotationPanel.x_deg_var.set("0")
            GUI.RotationPanel.y_deg_var.set("0")
            GUI.RotationPanel.z_deg_var.set("0")           

        def rotate_mesh():
            mesh = GUI.SaveResetPanel.get_mesh()
            if mesh:  
                try:
                    dg_x, dg_y, dg_z = float(GUI.RotationPanel.x_deg_var.get()), \
                                       float(GUI.RotationPanel.y_deg_var.get()), \
                                       float(GUI.RotationPanel.z_deg_var.get())
                    GUI.renderer.clear()
                    GUI.renderer.set_rotation_x_matrix(dg_x)
                    GUI.renderer.set_rotation_y_matrix(dg_y)
                    GUI.renderer.set_rotation_z_matrix(dg_z)
                    GUI.renderer.render(mesh)
                    GUI.renderer.canvas.update()
                except ValueError:
                    print("Enter a numeric value!")

    class ScalePanel:
        scale_factor_var = None
        
        def __init__(self, root, s_width, b_height, b_width, f_size):
            panel = tk.Frame(root)
            panel.pack()

            GUI.ScalePanel.scale_factor_var = tk.StringVar(root)
            GUI.ScalePanel.scale_factor_var.set("1.00")

            lb = tk.Label(panel, font=("", f_size, ""), text="Scale factor")
            lb.grid(row=0, column=1, padx=0, pady=0)
            
            scale = tk.Spinbox(panel, from_=0.1, to=4, increment=0.01, textvariable=GUI.ScalePanel.scale_factor_var,
                               width=s_width, font=("", f_size, ""))

            scale.grid(row=1, column=1)

            scale_b = tk.Button(panel, text="Scale", height=b_height, width=b_width,
                            font=("", f_size, ""), command=GUI.ScalePanel.scale_mesh)
            scale_b.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        def scale_mesh():
            mesh = GUI.SaveResetPanel.get_mesh()
            if mesh:  
                try:
                    s = float(GUI.ScalePanel.scale_factor_var.get())
                    GUI.renderer.clear()
                    GUI.renderer.set_scaling_matrix(s, s, s)
                    GUI.renderer.render(mesh)
                    GUI.renderer.canvas.update()
                except ValueError:
                    print("Enter a numeric value!")
                    
        def reset_variables():
            GUI.ScalePanel.scale_factor_var.set("1.00")

    class PointLightTransformationPanel:       
        x_var = None
        y_var = None
        z_var = None
        text_x = None
        text_y = None
        text_z = None
        text_translate = None
        
        X = 0
        Y = 0
        Z = 0
        DX = 0
        DY = 0
        DZ = 0
        
        def __init__(self, root, s_width, b_height, b_width, f_size):
            panel = tk.Frame(root)
            panel.pack()

            PLTP = GUI.PointLightTransformationPanel

            PLTP.X, PLTP.Y, PLTP.Z = GUI.renderer.get_light_position()
            PLTP.DX, PLTP.DY, PLTP.DZ = GUI.renderer.get_light_direction()   
            PLTP.x_var = tk.StringVar(root)
            PLTP.y_var = tk.StringVar(root)
            PLTP.z_var = tk.StringVar(root)

            PLTP.text_x = tk.StringVar(root)
            PLTP.text_y = tk.StringVar(root)
            PLTP.text_z = tk.StringVar(root)
            PLTP.text_translate = tk.StringVar(root)
  
            lb_x = tk.Label(panel, font=("", f_size, ""), textvariable=PLTP.text_x)
            lb_x.grid(row=0, column=0, padx=0, pady=5)
            lb_y = tk.Label(panel, font=("", f_size, ""), textvariable=PLTP.text_y)
            lb_y.grid(row=0, column=1, padx=0, pady=5)
            lb_z = tk.Label(panel, font=("", f_size, ""), textvariable=PLTP.text_z)
            lb_z.grid(row=0, column=2, padx=0, pady=5)

            box_x = tk.Spinbox(panel, from_=-4, to=4, increment=0.01, textvariable=PLTP.x_var,
                               width=s_width, font=("", f_size, ""))
            box_x.grid(row=1, column=0)
            
            box_y = tk.Spinbox(panel, from_=-4, to=4, increment=0.01, textvariable=PLTP.y_var,
                               width=s_width, font=("", f_size, ""))
            box_y.grid(row=1, column=1)

            box_z = tk.Spinbox(panel, from_=-4, to=4, increment=0.01, textvariable=PLTP.z_var,
                               width=s_width, font=("", f_size, ""))
            box_z.grid(row=1, column=2)
            
            translate = tk.Button(panel, height=b_height, textvariable=PLTP.text_translate,
                             font=("", f_size, ""), command=PLTP.translate_light)
            translate.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        def translate_light():
            mesh = GUI.SaveResetPanel.get_mesh()
            if mesh:  
                try:
                    PLTP = GUI.PointLightTransformationPanel
                    x, y, z = float(PLTP.x_var.get()), \
                              float(PLTP.y_var.get()), \
                              float(PLTP.z_var.get())
                    if GUI.Utility_Panel.lp_var.get():
                        GUI.renderer.set_light_position(x, y, z)
                    else:
                        GUI.renderer.set_light_direction(x, y, z)
                    GUI.renderer.clear()
                    GUI.renderer.render(mesh)
                    GUI.renderer.canvas.update()
                except ValueError:
                    print("Enter a numeric value!")
                    
        def reset_variables():
            PLTP = GUI.PointLightTransformationPanel
            GUI.renderer.set_light_position(PLTP.X, PLTP.Y, PLTP.Z)
            GUI.renderer.set_light_direction(PLTP.DX, PLTP.DY, PLTP.DZ)
            PLTP.change_mode(GUI.Utility_Panel.LP)

        def change_mode(light_position):
            PLTP = GUI.PointLightTransformationPanel
            if light_position:
                PLTP.text_x.set("x-axis")
                PLTP.text_y.set("y-axis")
                PLTP.text_z.set("z-axis")
                PLTP.x_var.set(str(PLTP.X))
                PLTP.y_var.set(str(PLTP.Y))
                PLTP.z_var.set(str(PLTP.Z))
                PLTP.text_translate.set("Set light position")
            else:             
                PLTP.text_x.set("x direction")
                PLTP.text_y.set("y direction")
                PLTP.text_z.set("z direction")
                PLTP.x_var.set(str(PLTP.DX))
                PLTP.y_var.set(str(PLTP.DY))
                PLTP.z_var.set(str(PLTP.DZ))
                PLTP.text_translate.set("Set light direction")

                
    class ReflectionConstantsPanel:
        ka_var = None
        kd_var = None
        ks_var = None
        h_var = None
        KA, KD, KS, H = 0, 0, 0, 0
        
        def __init__(self, root, s_width, b_height, b_width, f_size):
            panel = tk.Frame(root)
            panel.pack()

            RCP = GUI.ReflectionConstantsPanel
            
            RCP.KA = GUI.renderer.get_ambient_reflection()
            RCP.KD = GUI.renderer.get_diffuse_reflection()
            RCP.KS = GUI.renderer.get_specular_reflection()
            RCP.H = GUI.renderer.get_shininess()
            
            RCP.ka_var = tk.StringVar(root)
            RCP.kd_var = tk.StringVar(root)
            RCP.ks_var = tk.StringVar(root)
            RCP.h_var = tk.StringVar(root)
            
            RCP.reset_variables()

            lb_ka = tk.Label(panel, font=("", f_size, ""), text="ambient")
            lb_ka.grid(row=0, column=0, padx=0, pady=5)
            lb_kd = tk.Label(panel, font=("", f_size, ""), text="diffuse")
            lb_kd.grid(row=0, column=1, padx=0, pady=5)
            lb_ks = tk.Label(panel, font=("", f_size, ""), text="specular")
            lb_ks.grid(row=0, column=2, padx=0, pady=5)
            lb_h = tk.Label(panel, font=("", f_size, ""), text="shininess")
            lb_h.grid(row=0, column=3, padx=0, pady=5)

            box_ka = tk.Spinbox(panel, from_=0, to=2, increment=0.01, textvariable=RCP.ka_var,
                               width=s_width, font=("", f_size, ""))
            box_ka.grid(row=1, column=0)
            
            box_kd = tk.Spinbox(panel, from_=0, to=2, increment=0.01, textvariable=RCP.kd_var,
                               width=s_width, font=("", f_size, ""))
            box_kd.grid(row=1, column=1)

            box_ks = tk.Spinbox(panel, from_=0, to=2, increment=0.01, textvariable=RCP.ks_var,
                               width=s_width, font=("", f_size, ""))
            box_ks.grid(row=1, column=2)

            box_h = tk.Spinbox(panel, from_=0, to=1000, increment=10, textvariable=RCP.h_var,
                               width=s_width, font=("", f_size, ""))
            box_h.grid(row=1, column=3)
            
            lluminate = tk.Button(panel, text="Illuminate", height=b_height, width=b_width,
                             font=("", f_size, ""), command=RCP.change_illumination)
            lluminate.grid(row=2, column=0, columnspan=4, padx=10, pady=10)

        def change_illumination():
            mesh = GUI.SaveResetPanel.get_mesh()
            RCP = GUI.ReflectionConstantsPanel
            if mesh:  
                try:
                    ka, ks, kd, h = float(RCP.ka_var.get()), \
                                    float(RCP.ks_var.get()), \
                                    float(RCP.kd_var.get()), \
                                    float(RCP.h_var.get())
                    GUI.renderer.clear()
                    GUI.renderer.set_ambient_reflection(ka)
                    GUI.renderer.set_diffuse_reflection(kd)
                    GUI.renderer.set_specular_reflection(ks)
                    GUI.renderer.set_shininess(h)
                    GUI.renderer.render(mesh)
                    GUI.renderer.canvas.update()
                except ValueError:
                    print("Enter a numeric value!")
                    
        def reset_variables():
            RCP = GUI.ReflectionConstantsPanel      
            RCP.ka_var.set(str(RCP.KA))
            RCP.kd_var.set(str(RCP.KD))
            RCP.ks_var.set(str(RCP.KS))
            RCP.h_var.set(str(RCP.H))
            GUI.renderer.set_ambient_reflection(RCP.KA)
            GUI.renderer.set_diffuse_reflection(RCP.KD)
            GUI.renderer.set_specular_reflection(RCP.KS)
            GUI.renderer.set_shininess(RCP.H)

    class RGBPanel:
        r_var = None
        g_var = None
        b_var = None
        R, G, B = 0, 0, 0
        
        def __init__(self, root, s_width, b_height, b_width, f_size):
            panel = tk.Frame(root)
            panel.pack()
            
            GUI.RGBPanel.R, GUI.RGBPanel.G, GUI.RGBPanel.B = GUI.renderer.get_colour()   
            GUI.RGBPanel.r_var = tk.StringVar(root)
            GUI.RGBPanel.g_var = tk.StringVar(root)
            GUI.RGBPanel.b_var = tk.StringVar(root)
            GUI.RGBPanel.reset_variables()

            lb_x = tk.Label(panel, font=("", f_size, ""), text="r-value")
            lb_x.grid(row=0, column=0, padx=0, pady=5)
            lb_y = tk.Label(panel, font=("", f_size, ""), text="g-value")
            lb_y.grid(row=0, column=1, padx=0, pady=5)
            lb_z = tk.Label(panel, font=("", f_size, ""), text="b-value")
            lb_z.grid(row=0, column=2, padx=0, pady=5)

            box_r = tk.Spinbox(panel, from_=0, to=255, increment=1, textvariable=GUI.RGBPanel.r_var,
                               width=s_width, font=("", f_size, ""))
            box_r.grid(row=1, column=0)
            
            box_g = tk.Spinbox(panel, from_=0, to=255, increment=1, textvariable=GUI.RGBPanel.g_var,
                               width=s_width, font=("", f_size, ""))
            box_g.grid(row=1, column=1)

            box_b = tk.Spinbox(panel, from_=0, to=255, increment=1, textvariable=GUI.RGBPanel.b_var,
                               width=s_width, font=("", f_size, ""))
            box_b.grid(row=1, column=2)
            
            recolour = tk.Button(panel, text="Recolour", height=b_height, width=b_width,
                             font=("", f_size, ""), command=GUI.RGBPanel.change_colour)
            recolour.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        def change_colour():
            mesh = GUI.SaveResetPanel.get_mesh()
            if mesh:  
                try:
                    r, g, b = int(GUI.RGBPanel.r_var.get()), \
                              int(GUI.RGBPanel.g_var.get()), \
                              int(GUI.RGBPanel.b_var.get())
                    GUI.renderer.clear()
                    GUI.renderer.set_colour(r, g, b)
                    GUI.renderer.render(mesh)
                    GUI.renderer.canvas.update()
                except ValueError:
                    print("Enter a numeric value!")
                    
        def reset_variables():
            GUI.RGBPanel.r_var.set(str(GUI.RGBPanel.R))
            GUI.RGBPanel.g_var.set(str(GUI.RGBPanel.G))
            GUI.RGBPanel.b_var.set(str(GUI.RGBPanel.B))
            GUI.renderer.set_colour(GUI.RGBPanel.R, GUI.RGBPanel.G, GUI.RGBPanel.B)

    class Utility_Panel:
        bp_var = None
        wf_var = None
        lp_var = None
        BP = False
        WF = False
        LP = False
        
        def __init__(self, root, s_width, b_height, b_width, f_size):
            panel = tk.Frame(root)
            panel.pack()
            
            GUI.Utility_Panel.BP = GUI.renderer.blin_phong
            GUI.Utility_Panel.WF = GUI.renderer.wireframe
            GUI.Utility_Panel.LP = GUI.renderer.point_light

            GUI.Utility_Panel.bp_var = tk.BooleanVar(root)
            GUI.Utility_Panel.wf_var = tk.BooleanVar(root)
            GUI.Utility_Panel.lp_var = tk.BooleanVar(root)
            GUI.Utility_Panel.reset_variables()

            box_wf = tk.Checkbutton(panel, var=GUI.Utility_Panel.wf_var,
                                    command=GUI.Utility_Panel.wireframe_mode,
                                    font=("", f_size, ""), text="Wireframe")
            box_wf.grid(row=0, column=0)
            
            box_bp = tk.Checkbutton(panel, var=GUI.Utility_Panel.bp_var,
                                    command=GUI.Utility_Panel.blin_phnog_mode,
                                    font=("", f_size, ""), text="Blin-Phong")
            box_bp.grid(row=0, column=1)

            box_lp = tk.Checkbutton(panel, var=GUI.Utility_Panel.lp_var,
                                    command=GUI.Utility_Panel.light_mode,
                                    font=("", f_size, ""), text="Light position")
            box_lp.grid(row=0, column=2)

            GUI.PointLightTransformationPanel.change_mode(GUI.Utility_Panel.lp_var.get()) # initialises panel depending on mode

        def wireframe_mode():
            mesh = GUI.SaveResetPanel.get_mesh()
            if mesh:  
                GUI.renderer.wireframe = GUI.Utility_Panel.wf_var.get()
                GUI.renderer.clear()
                GUI.renderer.render(mesh)
                GUI.renderer.canvas.update()
            else:
                GUI.Utility_Panel.wf_var.set(GUI.Utility_Panel.WF)

        def blin_phnog_mode():
            mesh = GUI.SaveResetPanel.get_mesh()
            if mesh:  
                GUI.renderer.blin_phong = GUI.Utility_Panel.bp_var.get()
                GUI.renderer.clear()
                GUI.renderer.render(mesh)
                GUI.renderer.canvas.update()
            else:
                GUI.Utility_Panel.bp_var.set(GUI.Utility_Panel.BP)

        def light_mode():
            mesh = GUI.SaveResetPanel.get_mesh()
            if mesh:
                light_position = GUI.Utility_Panel.lp_var.get()
                GUI.PointLightTransformationPanel.change_mode(light_position)
                GUI.renderer.point_light = light_position
                GUI.renderer.clear()
                GUI.renderer.render(mesh)
                GUI.renderer.canvas.update()
            else:
                GUI.Utility_Panel.lp_var.set(GUI.Utility_Panel.LP)
                    
        def reset_variables():
            GUI.Utility_Panel.bp_var.set(GUI.Utility_Panel.BP)
            GUI.Utility_Panel.wf_var.set(GUI.Utility_Panel.WF)
            GUI.Utility_Panel.lp_var.set(GUI.Utility_Panel.LP)
            GUI.renderer.wireframe = GUI.Utility_Panel.WF
            GUI.renderer.blin_phong = GUI.Utility_Panel.BP
            GUI.renderer.point_light = GUI.Utility_Panel.LP
            GUI.PointLightTransformationPanel.reset_variables()

    renderer = None
    mesh = None
            
    def __init__(self):
        self.r_w = 800
        self.r_h = 800
        self.f_w = 200
        
        self.root = tk.Tk()
        self.root.title("Project")

        GUI.renderer = Renderer(self.root, self.r_w, self.r_h)
        
        panel = tk.Frame(self.root, width=self.f_w, height=self.r_h)
        panel.pack(side=tk.RIGHT)
        
        self.SaveResetPanel(panel, 1, 10, 16)
        self.TranslationPanel(panel, 4, 1, 10, 16)
        self.RotationPanel(panel, 4, 1, 10, 16)
        self.ScalePanel(panel, 4, 1, 10, 16)
        self.PointLightTransformationPanel(panel, 4, 1, 10, 16)
        self.RGBPanel(panel, 4, 1, 10, 16)
        self.ReflectionConstantsPanel(panel, 4, 1, 10, 16)
        self.Utility_Panel(panel, 4, 1, 10, 16)

        self.root.mainloop()
                
if __name__ == "__main__":
    a = GUI()
