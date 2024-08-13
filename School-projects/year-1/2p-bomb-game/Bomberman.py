import tkinter
import random
import time

class Menu:
    def __init__(self):
        self.hra_okno = tkinter.Tk()
        self.hra_okno.resizable(width=False, height=False)
        self.hra_okno.wm_attributes("-topmost", 1)
        self.hra_okno.title("Bomberman")
        self.hra_okno.minsize(480, 480)
        self.tlacidla = {}
        self.kanvas_okno = {}

        # vytvori canvas plochu a posle ju ostatnym triedam ako triedny atribut
        self.canvas = Hra.canvas = Hra.Vrchol.canvas = Hrac.canvas = Bomba.canvas = Vybuch.canvas = tkinter.Canvas(width=480, height=480, background="black")
        self.canvas.pack()
        self.hra = None
        self.nastavenie_ovladania = {"hore1" : "____", "dole1" : "____", "vpravo1" : "____",
                                        "vlavo1" : "____", "bomba1" : "____", "hore2" : "____",
                                        "dole2" : "____", "vpravo2" : "____","vlavo2" : "____",
                                        "bomba2" : "____"}

        # precita zo suborov vsetky obrazky a pouklada ich ako triedne atributy
        self.mapa_schema = {f"mapa{i}" : tkinter.PhotoImage(file=f"grafika/mapy/schema_mapa_{i}.png") for i in range(1, 7)}
        self.pozadie = tkinter.PhotoImage(file=f"grafika/pozadie/pozadie2.png")
        
        Hra.policko_sprite = {"skatula" : tkinter.PhotoImage(file="grafika/policka/skatula.png"),
                              "kamen" : tkinter.PhotoImage(file="grafika/policka/kamen.png"),
                              "trava" : tkinter.PhotoImage(file="grafika/policka/trava.png"),
                              "hlina" : tkinter.PhotoImage(file="grafika/policka/hlina.jpg"),
                              "kov" : tkinter.PhotoImage(file="grafika/policka/kov.png"),
                              "kov stena" : tkinter.PhotoImage(file="grafika/policka/kovova_skatula.jpg"),
                              "tehla" : tkinter.PhotoImage(file="grafika/policka/tehla.jpg"),
                              "zemina" : tkinter.PhotoImage(file="grafika/policka/zemina.jpg")}
        Hra.vylepsenia_sprite = {"plus_bomba" : tkinter.PhotoImage(file="grafika/vylepsenia/vylepsenie_1.png"),
                                "plus_vybuch" : tkinter.PhotoImage(file="grafika/vylepsenia/vylepsenie_2.png")}    
       
        Bomba.bomba_sprite = [tkinter.PhotoImage(file=f"grafika/bomba/bomba{i}.png") for i in range(4)]
        
        Vybuch.ohen_sprite[("M", 0)] = [tkinter.PhotoImage(file=f"grafika/plamen/ohenM1_{i}.png") for i in range(7)]
        Vybuch.ohen_sprite[("M", 1)] = [tkinter.PhotoImage(file=f"grafika/plamen/ohenM2_{i}.png") for i in range(7)]
        Vybuch.ohen_sprite[("R", 0)] = [tkinter.PhotoImage(file=f"grafika/plamen/ohenR0_{i}.png") for i in range(7)]
        Vybuch.ohen_sprite[("R", 1)] = [tkinter.PhotoImage(file=f"grafika/plamen/ohenR1_{i}.png") for i in range(7)]
        Vybuch.ohen_sprite[("L", 0)] = [tkinter.PhotoImage(file=f"grafika/plamen/ohenL0_{i}.png") for i in range(7)]
        Vybuch.ohen_sprite[("L", 1)] = [tkinter.PhotoImage(file=f"grafika/plamen/ohenL1_{i}.png") for i in range(7)]
        Vybuch.ohen_sprite[("U", 0)] = [tkinter.PhotoImage(file=f"grafika/plamen/ohenU0_{i}.png") for i in range(7)]
        Vybuch.ohen_sprite[("U", 1)] = [tkinter.PhotoImage(file=f"grafika/plamen/ohenU1_{i}.png") for i in range(7)]
        Vybuch.ohen_sprite[("D", 0)] = [tkinter.PhotoImage(file=f"grafika/plamen/ohenD0_{i}.png") for i in range(7)]
        Vybuch.ohen_sprite[("D", 1)] = [tkinter.PhotoImage(file=f"grafika/plamen/ohenD1_{i}.png") for i in range(7)]
            
        Hrac.hrac_1_sprite["D"] = [tkinter.PhotoImage(file=f"grafika/hrac/hrac1_{i}.png") for i in range(4)]
        Hrac.hrac_1_sprite["R"] = [tkinter.PhotoImage(file=f"grafika/hrac/hrac1_{i}.png") for i in range(4, 8)]
        Hrac.hrac_1_sprite["U"] = [tkinter.PhotoImage(file=f"grafika/hrac/hrac1_{i}.png") for i in range(8, 12)]  
        Hrac.hrac_1_sprite["L"] = [tkinter.PhotoImage(file=f"grafika/hrac/hrac1_{i}.png") for i in range(12, 16)]
        Hrac.hrac_1_sprite["X"] = [tkinter.PhotoImage(file=f"grafika/hrac/hrac1_16.png")]

        Hrac.hrac_2_sprite["D"] = [tkinter.PhotoImage(file=f"grafika/hrac/hrac2_{i}.png") for i in range(4)]
        Hrac.hrac_2_sprite["R"] = [tkinter.PhotoImage(file=f"grafika/hrac/hrac2_{i}.png") for i in range(4, 8)]
        Hrac.hrac_2_sprite["U"] = [tkinter.PhotoImage(file=f"grafika/hrac/hrac2_{i}.png") for i in range(8, 12)]  
        Hrac.hrac_2_sprite["L"] = [tkinter.PhotoImage(file=f"grafika/hrac/hrac2_{i}.png") for i in range(12, 16)]
        Hrac.hrac_2_sprite["X"] = [tkinter.PhotoImage(file=f"grafika/hrac/hrac2_16.png")]       

        self.vytvor_tlacidla_menu()
        self.canvas.bind_all('<KeyPress>', self.stlac_klaves)
        self.vypni_vsetky_prepinace()             
        self.vykresli_hlavne_menu()

        tkinter.mainloop()

    def vytvor_tlacidla_menu(self): # funkcia vytvori vsetky tlacidla na ktore moze uzivatel kliknut, metoda len inicializuje tlacidla, nevykresli ich
        # vytvori tlacidlo spat
        self.tlacidla["spat"] = Hra.tlacidlo_spat= tkinter.Button(self.canvas, text="X", 
                                                     relief="ridge", bg="red2", bd=4,
                                                     activebackground="red4", font="arial 16",
                                                     command=self.vykresli_hlavne_menu)
        
        # vytvori tlacidla hlavneho menu
        self.tlacidla["start1"] = tkinter.Button(self.canvas, text="Hra pre jedného hráča",
                                                     relief="ridge", bg="SteelBlue1", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=self.vypis_ponuku_1_hrac)
            
        self.tlacidla["start2"] = tkinter.Button(self.canvas, text="Hra pre dvoch hráčov",
                                                     relief="ridge", bg="SteelBlue1", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=self.vypis_ponuku_2_hraci)
            
        self.tlacidla["nastavenia"] = tkinter.Button(self.canvas, text="Nastavenia hry",
                                                         relief="ridge", bg="SteelBlue1", bd=4,
                                                         activebackground="yellow", font="arial 12",
                                                         command=self.vykresli_nastavenia)
        
        # vytvorenie tlacidiel nastavenia
        self.tlacidla["menej"] = tkinter.Button(self.canvas, text="<", 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 16",
                                                     command=self.uber_pocet_vyhier)

        self.tlacidla["viac"] = tkinter.Button(self.canvas, text=">", 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 16",
                                                     command=self.pridaj_pocet_vyhier)

        self.tlacidla["uloz"] = tkinter.Button(self.canvas, text="Ulož", 
                                                     relief="ridge", bg="pale green", bd=4,
                                                     activebackground="spring green", font="arial 14",
                                                     command=self.zapis_nastavenia)
        
        self.tlacidla["hore1"] = tkinter.Button(self.canvas, text=self.nastavenie_ovladania["hore1"], 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=lambda: self.prepni_tlacidlo("hore1"))
        
        self.tlacidla["hore2"] = tkinter.Button(self.canvas, text=self.nastavenie_ovladania["hore2"], 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=lambda: self.prepni_tlacidlo("hore2"))

        self.tlacidla["dole1"] = tkinter.Button(self.canvas, text=self.nastavenie_ovladania["dole1"], 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=lambda: self.prepni_tlacidlo("dole1"))
        
        self.tlacidla["dole2"] = tkinter.Button(self.canvas, text=self.nastavenie_ovladania["dole2"], 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=lambda: self.prepni_tlacidlo("dole2"))

        self.tlacidla["vpravo1"] = tkinter.Button(self.canvas, text=self.nastavenie_ovladania["vpravo1"], 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=lambda: self.prepni_tlacidlo("vpravo1"))
        
        self.tlacidla["vpravo2"] = tkinter.Button(self.canvas, text=self.nastavenie_ovladania["vpravo2"], 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=lambda: self.prepni_tlacidlo("vpravo2"))

        self.tlacidla["vlavo1"] = tkinter.Button(self.canvas, text=self.nastavenie_ovladania["vlavo1"], 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=lambda: self.prepni_tlacidlo("vlavo1"))
        
        self.tlacidla["vlavo2"] = tkinter.Button(self.canvas, text=self.nastavenie_ovladania["vlavo2"], 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=lambda: self.prepni_tlacidlo("vlavo2"))
        
        self.tlacidla["bomba1"] = tkinter.Button(self.canvas, text=self.nastavenie_ovladania["bomba1"], 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=lambda: self.prepni_tlacidlo("bomba1"))
        
        self.tlacidla["bomba2"] = tkinter.Button(self.canvas, text=self.nastavenie_ovladania["bomba2"], 
                                                     relief="ridge", bg="gold", bd=4,
                                                     activebackground="yellow2", font="arial 12",
                                                     command=lambda: self.prepni_tlacidlo("bomba2"))
        
        # vytvori tlacidla / jednotlive moznosti s obrazkom
        self.tlacidla["mapa1"] = tkinter.Button(self.canvas, image=self.mapa_schema["mapa1"],
                                                     relief="ridge", bg="SteelBlue1", bd=4,
                                                     activebackground="yellow2",
                                                     command=lambda: self.zapni_hru_mapa(1))
        self.tlacidla["mapa2"] = tkinter.Button(self.canvas, image=self.mapa_schema["mapa2"],
                                                     relief="ridge", bg="SteelBlue1", bd=4,
                                                     activebackground="yellow2",
                                                     command=lambda: self.zapni_hru_mapa(2))
        self.tlacidla["mapa3"] = tkinter.Button(self.canvas, image=self.mapa_schema["mapa3"],
                                                     relief="ridge", bg="SteelBlue1", bd=4,
                                                     activebackground="yellow2",
                                                     command=lambda: self.zapni_hru_mapa(3))
        self.tlacidla["mapa4"] = tkinter.Button(self.canvas, image=self.mapa_schema["mapa4"],
                                                     relief="ridge", bg="SteelBlue1", bd=4,
                                                     activebackground="yellow2",
                                                     command=lambda: self.zapni_hru_mapa(4))
        self.tlacidla["mapa5"] = tkinter.Button(self.canvas, image=self.mapa_schema["mapa5"],
                                                     relief="ridge", bg="SteelBlue1", bd=4,
                                                     activebackground="yellow2",
                                                     command=lambda: self.zapni_hru_mapa(5))
        self.tlacidla["mapa6"] = tkinter.Button(self.canvas, image=self.mapa_schema["mapa6"],
                                                     relief="ridge", bg="SteelBlue1", bd=4,
                                                     activebackground="yellow2",
                                                     command=lambda: self.zapni_hru_mapa(6))
    
    def precitaj_nastavenia(self): # funkcia precita z textoveho suboru pocet vyhier a ovladanie postav
        with open("nastavenia.txt") as nastavenia:
            self.pocet_vyhier = int(nastavenia.readline().strip())
            for instrukcia in self.nastavenie_ovladania:
                self.nastavenie_ovladania[instrukcia] = nastavenia.readline().strip()

    def zapis_nastavenia(self): # funkcia zapise do textoveho suboru pocet vyhier a ovladanie postav
        with open("nastavenia.txt", "w") as nastavenia:
            print(self.pocet_vyhier, file=nastavenia)
            for instrukcia in ("hore1", "dole1", "vpravo1", "vlavo1",
                               "bomba1", "hore2","dole2", "vpravo2",
                               "vlavo2", "bomba2"):
                print(self.nastavenie_ovladania[instrukcia], file=nastavenia)               
                
    def stlac_klaves(self, event): # event nastavi posledne stlacenu instrukciu podla uzivatelskeho vstupu
        klaves = event.keysym
        if klaves not in self.nastavenie_ovladania.values():
            if self.prepinac_upravovanych_tlacidiel["hore1"]:
                self.nastavenie_ovladania["hore1"] = klaves
                self.tlacidla["hore1"].config(text=klaves)
            elif self.prepinac_upravovanych_tlacidiel["hore2"]:
                self.nastavenie_ovladania["hore2"] = klaves
                self.tlacidla["hore2"].config(text=klaves)
            elif self.prepinac_upravovanych_tlacidiel["dole1"]:
                self.nastavenie_ovladania["dole1"] = klaves
                self.tlacidla["dole1"].config(text=klaves)
            elif self.prepinac_upravovanych_tlacidiel["dole2"]:
                self.nastavenie_ovladania["dole2"] = klaves
                self.tlacidla["dole2"].config(text=klaves)
            elif self.prepinac_upravovanych_tlacidiel["vpravo1"]:
                self.nastavenie_ovladania["vpravo1"] = klaves
                self.tlacidla["vpravo1"].config(text=klaves)
            elif self.prepinac_upravovanych_tlacidiel["vpravo2"]:
                self.nastavenie_ovladania["vpravo2"] = klaves
                self.tlacidla["vpravo2"].config(text=klaves)
            elif self.prepinac_upravovanych_tlacidiel["vlavo1"]:
                self.nastavenie_ovladania["vlavo1"] = klaves
                self.tlacidla["vlavo1"].config(text=klaves)
            elif self.prepinac_upravovanych_tlacidiel["vlavo2"]:
                self.nastavenie_ovladania["vlavo2"] = klaves
                self.tlacidla["vlavo2"].config(text=klaves)
            elif self.prepinac_upravovanych_tlacidiel["bomba1"]:
                self.nastavenie_ovladania["bomba1"] = klaves
                self.tlacidla["bomba1"].config(text=klaves)
            elif self.prepinac_upravovanych_tlacidiel["bomba2"]:
                self.nastavenie_ovladania["bomba2"] = klaves
                self.tlacidla["bomba2"].config(text=klaves)
        self.vypni_vsetky_prepinace()

    def vypni_vsetky_prepinace(self): # vypne vsetky tlacidla ktore bolo mozne stlacit, teda neocakava dalsi vstup od uzivatela
        self.canvas.unbind_all(self.stlac_klaves)
        self.prepinac_upravovanych_tlacidiel = {"hore1" : False, "dole1" : False, "vpravo1" : False,
                                                "vlavo1" : False, "bomba1" : False, "hore2" : False,
                                                "dole2" : False, "vpravo2" : False,"vlavo2" : False,
                                                "bomba2" : False}
        
        for tlacidlo, klaves in self.nastavenie_ovladania.items():
            self.tlacidla[tlacidlo].config(text=klaves)

    def prepni_tlacidlo(self, tlacidlo): # funkcia umozni zapisovat do posledne kliknuteho tlacidlo, nastavenia ocakavaju vstup od uzivatela
        self.vypni_vsetky_prepinace()
        self.canvas.bind_all('<KeyPress>', self.stlac_klaves)
        self.tlacidla[tlacidlo].config(text=">_____<")
        self.prepinac_upravovanych_tlacidiel[tlacidlo] = True

    def uber_pocet_vyhier(self): # funkcia uberie pocet vyhier potrebnych na vyhranie hry
        if self.pocet_vyhier > 1:
            self.pocet_vyhier -= 1
            self.canvas.itemconfig(self.zobraz_pocet_vyhier, text=f"{self.pocet_vyhier}")
            self.canvas.update()
            
    def pridaj_pocet_vyhier(self): # funkcia prida pocet vyhier potrebnych na vyhranie hry
        if self.pocet_vyhier < 10:
            self.pocet_vyhier += 1
            self.canvas.itemconfig(self.zobraz_pocet_vyhier, text=f"{self.pocet_vyhier}")
            self.canvas.update()
    
    def vykresli_hlavne_menu(self):     
        if self.hra: # v tomto pirpade sa hra vrati do ponuky
            self.hra.hra_zapnuta = False
            self.hra = None

        else: # v tomto pripade sa hra vrati do hlavneho menu          
            self.canvas.unbind_all(self.stlac_klaves)
            self.canvas.delete("all")
            self.canvas.create_image(242, 242, image=self.pozadie)
            
            self.kanvas_okno["start1"] = self.canvas.create_window(240, 300, width=190, height=50,
                                                                   window=self.tlacidla["start1"])
            self.kanvas_okno["start2"] = self.canvas.create_window(240, 360, width=190, height=50,
                                                                   window=self.tlacidla["start2"])
            self.kanvas_okno["nastavenia"] = self.canvas.create_window(240, 420, width=190, height=50,
                                                                       window=self.tlacidla["nastavenia"])

    def vykresli_nastavenia(self):
        self.canvas.delete("all") # vymaze predoslu ponuku spolu s tlacidlami
        self.precitaj_nastavenia() # precita subor a ulozi ho do self.nastavenie_ovladania
        for tlacidlo, klaves in self.nastavenie_ovladania.items():
            self.tlacidla[tlacidlo].config(text=klaves)

        # nakresli hlavicku, okno ponuky, vypise pocet vyhier potrebnych na vyhranie jednej hry
        self.canvas.create_rectangle(40, 40, 440, 80, fill="white", outline="deep sky blue", width=3)
        self.canvas.create_rectangle(40, 80, 440, 440, fill="sky blue", outline="deep sky blue", width=3)
        self.canvas.create_rectangle(250, 90, 290, 130, fill="white", outline="deep sky blue", width=3)
        self.zobraz_pocet_vyhier = self.canvas.create_text(270, 110, text=f"{self.pocet_vyhier}", font="arial 12")

        # vlozi tlacidla do plochy
        self.kanvas_okno["spat"] = self.canvas.create_window(420, 60, width=40, height=40,
                                                                   window=self.tlacidla["spat"])
        self.kanvas_okno["uloz"] = self.canvas.create_window(370, 60, width=60, height=40,
                                                                   window=self.tlacidla["uloz"])
        self.kanvas_okno["menej"] = self.canvas.create_window(220, 110, width=40, height=40,
                                                                   window=self.tlacidla["menej"])
        self.kanvas_okno["viac"] = self.canvas.create_window(320, 110, width=40, height=40,
                                                                   window=self.tlacidla["viac"])
        self.kanvas_okno["hore1"] = self.canvas.create_window(220, 210, width=90, height=40,
                                                                   window=self.tlacidla["hore1"])
        self.kanvas_okno["hore2"] = self.canvas.create_window(320, 210, width=90, height=40,
                                                                   window=self.tlacidla["hore2"])
        self.kanvas_okno["dole1"] = self.canvas.create_window(220, 260, width=90, height=40,
                                                                   window=self.tlacidla["dole1"])
        self.kanvas_okno["dole2"] = self.canvas.create_window(320, 260, width=90, height=40,
                                                                   window=self.tlacidla["dole2"])
        self.kanvas_okno["vpravo1"] = self.canvas.create_window(220, 310, width=90, height=40,
                                                                   window=self.tlacidla["vpravo1"])
        self.kanvas_okno["vpravo2"] = self.canvas.create_window(320, 310, width=90, height=40,
                                                                   window=self.tlacidla["vpravo2"])
        self.kanvas_okno["vlavo1"] = self.canvas.create_window(220, 360, width=90, height=40,
                                                                   window=self.tlacidla["vlavo1"])
        self.kanvas_okno["vlavo2"] = self.canvas.create_window(320, 360, width=90, height=40,
                                                                   window=self.tlacidla["vlavo2"])
        self.kanvas_okno["bomba1"] = self.canvas.create_window(220, 410, width=90, height=40,
                                                                   window=self.tlacidla["bomba1"])
        self.kanvas_okno["bomba2"] = self.canvas.create_window(320, 410, width=90, height=40,
                                                                   window=self.tlacidla["bomba2"])
        # vypise text
        self.canvas.create_text(190, 60, text="Nastavenia hry", font="arial 16")
        self.canvas.create_text(50, 110, text="Počet výhier:", font="arial 12", anchor="w")
        self.canvas.create_text(50, 160, text="ovládanie:", font="arial 12", anchor="w")
        self.canvas.create_text(200, 160, text="hráč 1", font="arial 12", anchor="w")
        self.canvas.create_text(300, 160, text="hráč 2", font="arial 12", anchor="w")
        self.canvas.create_text(50, 210, text="pohyb nahor:", font="arial 12", anchor="w")
        self.canvas.create_text(50, 260, text="pohyb nadol:", font="arial 12", anchor="w")
        self.canvas.create_text(50, 310, text="pohyb vpravo:", font="arial 12", anchor="w")
        self.canvas.create_text(50, 360, text="pohyb vľavo:", font="arial 12", anchor="w")
        self.canvas.create_text(50, 410, text="polož bombu:", font="arial 12", anchor="w")

    def vykresli_vyber_mapy(self):
        # vytvori okno
        self.canvas.create_rectangle(40, 40, 440, 80, fill="white", outline="deep sky blue", width=3)
        self.canvas.create_rectangle(40, 80, 440, 440, fill="sky blue", outline="deep sky blue", width=3)

        #vykresli tlacidla v ploche
        self.kanvas_okno["mapa1"] = self.canvas.create_window(117, 186, width=95, height=82,
                                                                   window=self.tlacidla["mapa1"])

        self.kanvas_okno["mapa2"] = self.canvas.create_window(240, 186, width=95, height=82,
                                                                   window=self.tlacidla["mapa2"])

        self.kanvas_okno["mapa3"] = self.canvas.create_window(363, 186, width=95, height=82,
                                                                   window=self.tlacidla["mapa3"])

        self.kanvas_okno["mapa4"] = self.canvas.create_window(117, 333, width=95, height=82,
                                                                   window=self.tlacidla["mapa4"])

        self.kanvas_okno["mapa5"] = self.canvas.create_window(240, 333, width=95, height=82,
                                                                   window=self.tlacidla["mapa5"])

        self.kanvas_okno["mapa6"] = self.canvas.create_window(363, 333, width=95, height=82,
                                                                   window=self.tlacidla["mapa6"])

        #vykresli tlacidlo spat
        self.kanvas_okno["spat"] = self.canvas.create_window(420, 60, width=40, height=40,
                                                                   window=self.tlacidla["spat"])

    def zapni_hru_mapa(self, identifikator):
        self.canvas.delete("all")
        if self.dvaja_hraci:
            self.hra = Hra(f"mapy/mapa_{identifikator}.txt")
            self.hra.start()
            self.hra = None
            self.vypis_ponuku_2_hraci()
        else:
            self.hra = Hra(f"mapy/mapa_{identifikator}.txt", False)
            self.hra.start(False)
            self.hra = None
            self.vypis_ponuku_1_hrac()
        
    def vypis_ponuku_2_hraci(self):
        self.canvas.delete("all")
        self.vykresli_vyber_mapy()
        #vypise nazov ponuky
        self.canvas.create_text(210, 60, text="Výber mapy (dvaja hráči):", font="arial 16")
        self.dvaja_hraci = True
        
    def vypis_ponuku_1_hrac(self):
        self.canvas.delete("all")
        self.vykresli_vyber_mapy()
        #vypise nazov ponuky
        self.canvas.create_text(210, 60, text="Výber mapy (jeden hráč):", font="arial 16")
        self.dvaja_hraci = False


class Vybuch:
    policka_s_vybuchom = None
    canvas = None
    id_skatul = {}
    mapa = None
    ohen_sprite = {}

    def __init__(self, x, y, dlzka = 0): #metoda kontroluje co znicil vybuch, pripadne rozhoduje kolko spritov mozno vykreslit
        self.x, self.y = x, y
        self.dlzka = dlzka
        self.objekt = []
        self.faza = 0
        self.koniec_animacie = False
        
        prekazka = {"U" : False, "D" : False, "R" : False, "L" : False}
        vzdialenost = {"U" : 0, "D" : 0, "R" : 0, "L" : 0}
        self.objekt += [("M0", self.canvas.create_image(8 + x * 32, 16 + y * 32, image=self.ohen_sprite[("M", 0)][0])),
                        ("M1",self.canvas.create_image(24 + x * 32, 16 + y * 32, image=self.ohen_sprite[("M", 1)][0]))]
        self.vybuch = [(x, y)]
        self.mapa[y][x].vybuch = True
        for i in range(dlzka):
            if x - 1 - i > 0 :
                if self.mapa[y][x - 1 - i].hodnota in ("0", "B") and not prekazka["L"]:
                    self.objekt += [("L0", self.canvas.create_image(8 + (x - 1 - i) * 32, 16 + y * 32, image=self.ohen_sprite[("L", 0)][0])),
                                    ("L0", self.canvas.create_image(24 + (x - 1 - i) * 32, 16 + y * 32, image=self.ohen_sprite[("L", 0)][0]))]
                    vzdialenost["L"] += 1
                    self.vybuch.append((x - 1 - i, y))
                    self.mapa[y][x - 1 - i].vybuch = True
                    
                    if self.mapa[y][x - 1 - i].vylepsenie_typ:
                        self.mapa[y][x - 1 - i].vymaz_vylepsenie()
                    
                elif self.mapa[y][x - 1 - i].hodnota == "b" or self.mapa[y][x - 1 - i].hodnota == "x":
                    prekazka["L"] = True
                    
            if x + 1 + i < len(self.mapa[y]):
                if self.mapa[y][x + 1 + i].hodnota in ("0", "B") and not prekazka["R"]: 
                    self.objekt += [("R0", self.canvas.create_image(8 + (x + 1 + i) * 32, 16 + y * 32, image=self.ohen_sprite[("R", 0)][0])),
                                    ("R0", self.canvas.create_image(24 + (x + 1 + i) * 32, 16 + y * 32, image=self.ohen_sprite[("R", 0)][0]))]
                    vzdialenost["R"] += 1
                    self.vybuch.append((x + 1 + i, y))
                    self.mapa[y][x + 1 + i].vybuch = True

                    if self.mapa[y][x + 1 + i].vylepsenie_typ:
                        self.mapa[y][x + 1 + i].vymaz_vylepsenie()
                    
                elif self.mapa[y][x + 1 + i].hodnota == "b" or self.mapa[y][x + 1 + i].hodnota == "x":
                    prekazka["R"] = True
                    
            if len(self.mapa) > y - 1 - i:
                if self.mapa[y - 1 - i][x].hodnota in ("0", "B") and not prekazka["U"]:
                    self.objekt += [("U0", self.canvas.create_image(16 + x * 32, 8 + (y - 1 - i) * 32, image=self.ohen_sprite[("U", 0)][0])),
                                    ("U0", self.canvas.create_image(16 + x * 32, 24 + (y - 1 - i) * 32, image=self.ohen_sprite[("U", 0)][0]))]
                    vzdialenost["U"] += 1
                    self.vybuch.append((x, y - 1 - i))
                    self.mapa[y - 1 - i][x].vybuch = True

                    if self.mapa[y - 1 - i][x].vylepsenie_typ:
                        self.mapa[y - 1 - i][x].vymaz_vylepsenie()
                    
                elif self.mapa[y - 1 - i][x].hodnota == "b" or self.mapa[y - 1 - i][x].hodnota == "x":
                    prekazka["U"] = True
                    
            if y + 1 + i < len(self.mapa):
                if self.mapa[y + 1 + i][x].hodnota in ("0", "B") and not prekazka["D"]:
                    self.objekt += [("D0", self.canvas.create_image(16 + x * 32, 8 + (y + 1 + i) * 32, image=self.ohen_sprite[("D", 0)][0])),
                                    ("D0", self.canvas.create_image(16 + x * 32, 24 + (y + 1 + i) * 32, image=self.ohen_sprite[("D", 0)][0]))]
                    vzdialenost["D"] += 1
                    self.vybuch.append((x, y + 1 + i))
                    self.mapa[y + 1 + i][x].vybuch = True

                    if self.mapa[y + 1 + i][x].vylepsenie_typ:
                        self.mapa[y + 1 + i][x].vymaz_vylepsenie()
                        
                elif self.mapa[y + 1 + i][x].hodnota == "b" or self.mapa[y + 1 + i][x].hodnota == "x": 
                    prekazka["D"] = True


        if x - 1 - vzdialenost["L"] > 0:
            if self.mapa[y][x - 1 - vzdialenost["L"]].hodnota in ("0", "B") and self.mapa[y][x - 1 - vzdialenost["L"]].hodnota != "x":
                self.objekt += [("L1", self.canvas.create_image(16 + (x - 1 - vzdialenost["L"]) * 32, 16 + y * 32, image=self.ohen_sprite[("L", 1)][0]))] # hlava vybuchu smerom vlavo
                self.vybuch.append((x - 1 - vzdialenost["L"], y))
                self.mapa[y][x - 1 - vzdialenost["L"]].vybuch = True # nastavi na policku vybuch
                self.mapa[y][x - 1 - vzdialenost["L"]].hodnota = "0" # vymaze skatulu na policku
                if self.mapa[y][x - 1 - vzdialenost["L"]].vylepsenie_typ:
                    self.mapa[y][x - 1 - vzdialenost["L"]].vymaz_vylepsenie()

            elif self.mapa[y][x - 1 - vzdialenost["L"]].hodnota == "b":
                skatula = self.id_skatul[(x - 1 - vzdialenost["L"], y)]
                self.vybuch.append((x - 1 - vzdialenost["L"], y))
                self.mapa[y][x - 1 - vzdialenost["L"]].vybuch = True # nastavi na policku vybuch               
                self.mapa[y][x - 1 - vzdialenost["L"]].hodnota = "0" # vymaze skatulu na policku
                self.prirad_suseda(self.mapa[y][x - 1 - vzdialenost["L"]])
                try: #v pripade ze kluc bol medzicasom odstraneny inou bombou
                    self.canvas.delete(self.id_skatul[(x - 1 - vzdialenost["L"], y)])
                    del self.id_skatul[(x - 1 - vzdialenost["L"], y)]
                except:
                    pass
                self.mapa[y][x - 1 - vzdialenost["L"]].pridaj_vylepsenie()
                self.mapa_grafika.append(self.mapa[y][x - 1 - vzdialenost["L"]].vylepsenie_objekt)
                self.objekt += [("L1", self.canvas.create_image(16 + (x - 1 - vzdialenost["L"]) * 32, 16 + y * 32, image=self.ohen_sprite[("L", 1)][0]))] # hlava vybuchu smerom vlavo
                
        if x + 1 + vzdialenost["R"] < len(self.mapa[y]):
            if self.mapa[y][x + 1 + vzdialenost["R"]].hodnota in ("0", "B") and self.mapa[y][x + 1 + vzdialenost["R"]].hodnota != "x":
                self.objekt += [("R1", self.canvas.create_image(16 + (x + 1 + vzdialenost["R"]) * 32, 16 + y * 32, image=self.ohen_sprite[("R", 1)][0]))] # hlava vybuchu smerom vpravo
                self.vybuch.append((x + 1 + vzdialenost["R"], y))
                self.mapa[y][x + 1 + vzdialenost["R"]].vybuch = True # nastavi na policku vybuch
                self.mapa[y][x + 1 + vzdialenost["R"]].hodnota = "0" # vymaze skatulu na policku
                if self.mapa[y][x + 1 + vzdialenost["R"]].vylepsenie_typ:
                    self.mapa[y][x + 1 + vzdialenost["R"]].vymaz_vylepsenie()

            elif self.mapa[y][x + 1 + vzdialenost["R"]].hodnota == "b":
                skatula = self.id_skatul[(x + 1 + vzdialenost["R"], y)]
                self.vybuch.append((x + 1 + vzdialenost["R"], y))
                self.mapa[y][x + 1 + vzdialenost["R"]].vybuch = True # nastavi na policku vybuch
                self.mapa[y][x + 1 + vzdialenost["R"]].hodnota = "0" # vymaze skatulu na policku
                self.prirad_suseda(self.mapa[y][x + 1 + vzdialenost["R"]])
                try: #v pripade ze kluc bol medzicasom odstraneny inou bombou
                    self.canvas.delete(self.id_skatul[(x + 1 + vzdialenost["R"], y)])
                    del self.id_skatul[(x + 1 + vzdialenost["R"], y)]
                except:
                    pass
                self.mapa[y][x + 1 + vzdialenost["R"]].pridaj_vylepsenie()
                self.mapa_grafika.append(self.mapa[y][x + 1 + vzdialenost["R"]].vylepsenie_objekt)
                self.objekt += [("R1", self.canvas.create_image(16 + (x + 1 + vzdialenost["R"]) * 32, 16 + y * 32, image=self.ohen_sprite[("R", 1)][0]))] # hlava vybuchu smerom vpravo              
            
        if y - 1 - vzdialenost["U"] > 0:
            if self.mapa[y - 1 - vzdialenost["U"]][x].hodnota in ("0", "B") and self.mapa[y - 1 - vzdialenost["U"]][x].hodnota != "x":
                self.objekt += [("U1", self.canvas.create_image(16 + x * 32, 16 + (y - 1 - vzdialenost["U"]) * 32, image=self.ohen_sprite[("U", 1)][0]))] # hlava vybuchu smerom nahor
                self.vybuch.append((x, y - 1 - vzdialenost["U"]))
                self.mapa[y - 1 - vzdialenost["U"]][x].vybuch = True # nastavi na policku vybuch
                self.mapa[y - 1 - vzdialenost["U"]][x].hodnota = "0" # vymaze skatulu na policku
                if self.mapa[y - 1 - vzdialenost["U"]][x].vylepsenie_typ:
                    self.mapa[y - 1 - vzdialenost["U"]][x].vymaz_vylepsenie()
                
            elif self.mapa[y - 1 - vzdialenost["U"]][x].hodnota == "b":
                skatula = self.id_skatul[(x, y - 1 - vzdialenost["U"])]
                self.vybuch.append((x, y - 1 - vzdialenost["U"]))
                self.mapa[y - 1 - vzdialenost["U"]][x].vybuch = True # nastavi na policku vybuch
                self.mapa[y - 1 - vzdialenost["U"]][x].hodnota = "0" # vymaze skatulu na policku
                self.prirad_suseda(self.mapa[y - 1 - vzdialenost["U"]][x])
                try: #v pripade ze kluc bol medzicasom odstraneny inou bombou
                    self.canvas.delete(self.id_skatul[(x, y - 1 - vzdialenost["U"])])
                    del self.id_skatul[(x, y - 1 - vzdialenost["U"])]
                except:
                    pass
                self.mapa[y - 1 - vzdialenost["U"]][x].pridaj_vylepsenie()
                self.mapa_grafika.append(self.mapa[y - 1 - vzdialenost["U"]][x].vylepsenie_objekt)
                self.objekt += [("U1", self.canvas.create_image(16 + x * 32, 16 + (y - 1 - vzdialenost["U"]) * 32, image=self.ohen_sprite[("U", 1)][0]))] # hlava vybuchu smerom nahor
                
        if y + 1 + vzdialenost["D"] < len(self.mapa):
            if self.mapa[y + 1 + vzdialenost["D"]][x].hodnota in ("0", "B") and self.mapa[y + 1 + vzdialenost["D"]][x].hodnota != "x":
                self.objekt += [("D1", self.canvas.create_image(16 + x * 32, 16 + (y + 1 + vzdialenost["D"]) * 32, image=self.ohen_sprite[("D", 1)][0]))] # hlava vybuchu smerom nadol
                self.vybuch.append((x, y + 1 + vzdialenost["D"]))
                self.mapa[y + 1 + vzdialenost["D"]][x].vybuch = True # nastavi na policku vybuch
                self.mapa[y + 1 + vzdialenost["D"]][x].hodnota = "0" # vymaze skatulu na policku
                if self.mapa[y + 1 + vzdialenost["D"]][x].vylepsenie_typ:
                    self.mapa[y + 1 + vzdialenost["D"]][x].vymaz_vylepsenie()

            elif self.mapa[y + 1 + vzdialenost["D"]][x].hodnota == "b":
                skatula = self.id_skatul[(x, y + 1 + vzdialenost["D"])]
                self.vybuch.append((x, y + 1 + vzdialenost["D"]))
                self.mapa[y + 1 + vzdialenost["D"]][x].vybuch = True # nastavi na policku vybuch
                self.mapa[y + 1 + vzdialenost["D"]][x].hodnota = "0" # vymaze skatulu na policku
                self.prirad_suseda(self.mapa[y + 1 + vzdialenost["D"]][x])
                try: #v pripade ze kluc bol medzicasom odstraneny inou bombou
                    self.canvas.delete(self.id_skatul[(x, y + 1 + vzdialenost["D"])])
                    del self.id_skatul[(x, y + 1 + vzdialenost["D"])]
                except:
                    pass
                self.mapa[y + 1 + vzdialenost["D"]][x].pridaj_vylepsenie()
                self.mapa_grafika.append(self.mapa[y + 1 + vzdialenost["D"]][x].vylepsenie_objekt)
                self.objekt += [("D1", self.canvas.create_image(16 + x * 32, 16 + (y + 1 + vzdialenost["D"]) * 32, image=self.ohen_sprite[("D", 1)][0]))] # hlava vybuchu smerom nadol
        self.policka_s_vybuchom.append(self.vybuch)

    def prirad_suseda(self, vrchol): # priradi vrcholom nad, pod, vpravo a vlavo, konkrety vrchol ak su prazdnym polickom
        x, y = vrchol.x, vrchol.y
        if x - 1 >= 0:
            if self.mapa[y][x - 1].hodnota == "0":
                self.mapa[y][x - 1].pridaj_hranu(vrchol)
                vrchol.pridaj_hranu(self.mapa[y][x - 1])

        if x + 1 < len(self.mapa[y]):
            if self.mapa[y][x + 1].hodnota == "0":
                self.mapa[y][x + 1].pridaj_hranu(vrchol)
                vrchol.pridaj_hranu(self.mapa[y][x + 1])

        if y - 1 >= 0:
            if self.mapa[y - 1][x].hodnota == "0":
                self.mapa[y - 1][x].pridaj_hranu(vrchol)
                vrchol.pridaj_hranu(self.mapa[y - 1][x])
                
        if y + 1 < len(self.mapa):
            if self.mapa[y + 1][x].hodnota == "0":
                self.mapa[y + 1][x].pridaj_hranu(vrchol)
                vrchol.pridaj_hranu(self.mapa[y + 1][x])
        
    def animuj(self): #metoda animuje jednotlive sprity vybuchu bomby      
        if self.faza + 1 == 7:
            self.koniec_animacie = True
            for x, y in self.vybuch:
                self.mapa[y][x].casovac = 0
                self.mapa[y][x].vybuch = False
                self.mapa[y][x].ohrozeny = False
            try:
                self.policka_s_vybuchom.remove(self.vybuch)
            except:
                pass
            for typ, idnt in self.objekt:
                self.canvas.delete(idnt)
                
        else:
            for x, y in self.vybuch: # v pripade ze inych vybuch skoncil a zmenil susedne policko
                self.mapa[y][x].vybuch = True
                self.mapa[y][x].ohrozeny = True
            self.faza = self.faza + 1
            for typ, idnt in self.objekt:
                if typ == "M0":
                    self.canvas.itemconfig(idnt, image=self.ohen_sprite[("M", 0)][self.faza])
                elif typ == "M1":
                    self.canvas.itemconfig(idnt, image=self.ohen_sprite[("M", 1)][self.faza])
                elif typ == "R0":
                    self.canvas.itemconfig(idnt, image=self.ohen_sprite[("R", 0)][self.faza])
                elif typ == "R1":
                    self.canvas.itemconfig(idnt, image=self.ohen_sprite[("R", 1)][self.faza])
                elif typ == "L0":
                    self.canvas.itemconfig(idnt, image=self.ohen_sprite[("L", 0)][self.faza])
                elif typ == "L1":
                    self.canvas.itemconfig(idnt, image=self.ohen_sprite[("L", 1)][self.faza])
                elif typ == "U0":
                    self.canvas.itemconfig(idnt, image=self.ohen_sprite[("U", 0)][self.faza])
                elif typ == "U1":
                    self.canvas.itemconfig(idnt, image=self.ohen_sprite[("U", 1)][self.faza])
                elif typ == "D0":
                    self.canvas.itemconfig(idnt, image=self.ohen_sprite[("D", 0)][self.faza])
                elif typ == "D1":
                    self.canvas.itemconfig(idnt, image=self.ohen_sprite[("D", 1)][self.faza])


class Bomba:
    bomby = None
    bomba_sprite = None
    canvas = None
    mapa = None
    vybuchy = None
    policka_s_vybuchom = None
    
    def __init__(self, x, y, idnt, dlzka=1): # umiestni do plochy bombu
        self.dlzka = dlzka
        self.vybuchni = False
        self.mapa[y][x].hodnota = "B"
        self.mapa[y][x].ohrozeny = True
        self.casovac = 0
        self.faza_animacie = 0
        self.majitel = idnt
        self.x, self.y = x, y
        self.objekt = self.canvas.create_image(16 + (x * 32), 16 + (y * 32), image=self.bomba_sprite[self.faza_animacie])
        self.urc_ohrozene_policka()

    def urc_ohrozene_policka(self): # urci na ktorych polickach hrozi vybuch
        prekazka = {"L" : False, "R" : False, "U" : False, "D" : False}
        self.mapa[self.y][self.x].casovac = self.casovac
        for i in range(self.dlzka + 1):
            if self.x + 1 + i < len(self.mapa[self.y]):
                if self.mapa[self.y][self.x + 1 + i].hodnota == "0" and not prekazka["D"]:
                    self.mapa[self.y][self.x + 1 + i].ohrozeny = True
                    self.mapa[self.y][self.x + 1 + i].casovac = self.casovac
                elif self.mapa[self.y][self.x + 1 + i].hodnota == "b":
                    prekazka["D"] = True
            if self.x - 1 - i >= 0:
                if self.mapa[self.y][self.x - 1 - i].hodnota == "0" and not prekazka["U"]:
                    self.mapa[self.y][self.x - 1 - i].ohrozeny = True
                    self.mapa[self.y][self.x - 1 - i].casovac = self.casovac
                elif self.mapa[self.y][self.x - 1 - i].hodnota == "b":
                    prekazka["U"] = True
            if self.y + 1 + i < len(self.mapa):
                if self.mapa[self.y + 1 + i][self.x].hodnota == "0" and not prekazka["R"]:
                    self.mapa[self.y + 1 + i][self.x].ohrozeny = True
                    self.mapa[self.y + 1 + i][self.x].casovac = self.casovac
                    if self.mapa[self.y + 1 + i][self.x].hodnota == "b":
                        prekazka["R"] = True
            if self.y - 1 - i >= 0:
                if self.mapa[self.y - 1 - i][self.x].hodnota == "0" and not prekazka["L"]:
                    self.mapa[self.y - 1 - i][self.x].ohrozeny = True
                    self.mapa[self.y - 1 - i][self.x].casovac = self.casovac
                elif self.mapa[self.y - 1 - i][self.x].hodnota == "b":
                    prekazka["L"] = True

    def animuj(self): # nafukovanie bomby spojene s odpocitavanim casovacu bomby
        for policka_s_vybuchom in self.policka_s_vybuchom:
            if (self.x, self.y) in policka_s_vybuchom:
                self.vybuch()
                break
        self.urc_ohrozene_policka()
        self.faza_animacie = ((self.faza_animacie + 1) % 8)
        self.casovac += 1
        self.canvas.itemconfig(self.objekt, image=self.bomba_sprite[self.faza_animacie//2])
        if self.casovac == 30:
            self.vybuch()

    def vybuch(self): # odstrani z mapy polozenu bombu a zaroven inicializuje jej vybuch
        self.vybuchni = True
        self.mapa[self.y][self.x].hodnota = "0"
        self.canvas.delete(self.objekt)
        self.vybuchy.append(Vybuch(self.x, self.y, self.dlzka))
        self.canvas.update()


class Hrac:
    mapa = None
    canvas = None
    hrac_1_sprite = {}
    hrac_2_sprite = {}
    
    def __init__(self, y, x, hrac1 = True):
        self.hrac = {}
        self.x, self.y = 16 + x * 32, 16 + y * 32 
        self.smer = None
        self.faza = 0
        self.pocet_bomb = 0
        self.dlzka_vybuchu = 0
        self.max_pocet_bomb = 1
        
        if hrac1:
            self.hrac_sprite = self.hrac_1_sprite
        else:
            self.hrac_sprite = self.hrac_2_sprite
        self.objekt = self.canvas.create_image(1 + 16 + x * 32, 1 + y * 32, image=self.hrac_sprite["U"][1])
                
    def zmen_smer(self, smer): # zmena smeru postavy hraca, podla vstupu
        self.smer = {"hore" : "U", "dole" : "D", "vlavo" : "L", "vpravo" : "R"}[smer]

    def zastav(self): # metoda vykresli spravny sprite ked hrac zastavi svoj pohyb
        self.faza = 1
        if self.smer == "U":
            self.canvas.itemconfig(self.objekt, image=self.hrac_sprite["U"][self.faza])
        elif self.smer == "D":
            self.canvas.itemconfig(self.objekt, image=self.hrac_sprite["D"][self.faza])
        elif self.smer == "R":
            self.canvas.itemconfig(self.objekt, image=self.hrac_sprite["R"][self.faza])
        elif self.smer == "L":
            self.canvas.itemconfig(self.objekt, image=self.hrac_sprite["L"][self.faza])
        self.faza = 0
        self.smer = None      
    
    def animuj(self): # metoda animuje/posuva postavu hraca a zaroven kontroluje ci sa postava moze posunut
        if self.smer == "U":
            self.canvas.itemconfig(self.objekt, image=self.hrac_sprite["U"][self.faza])
            if (self.y - 12) // 32 > 0: 
                if (self.mapa[(self.y - 12) // 32][self.x // 32].hodnota == "0") or \
                   ((self.mapa[self.y // 32][self.x // 32].hodnota == "B") and \
                    (self.mapa[(self.y - 32) // 32][self.x // 32].hodnota == "0")):
                    self.y -= 8
                    self.canvas.move(self.objekt, 0, -8)
        elif self.smer == "D":
            self.canvas.itemconfig(self.objekt, image=self.hrac_sprite["D"][self.faza])
            if (self.y + 12) // 32 < len(self.mapa):
                if self.mapa[(self.y + 12) // 32][self.x // 32].hodnota == "0" or \
                ((self.mapa[self.y // 32][self.x // 32].hodnota == "B") and \
                (self.mapa[(self.y + 32) // 32][self.x // 32].hodnota == "0")):
                    self.y += 8
                    self.canvas.move(self.objekt, 0, 8)
        elif self.smer == "R":
            self.canvas.itemconfig(self.objekt, image=self.hrac_sprite["R"][self.faza])
            if (self.x + 12) // 32 < len(self.mapa[(self.y + 12 )// 32]):
                if self.mapa[self.y // 32][(self.x + 8) // 32].hodnota == "0" or \
                   ((self.mapa[self.y // 32][self.x // 32].hodnota == "B") and \
                    (self.mapa[self.y // 32][(self.x + 32) // 32].hodnota == "0")):
                    self.x += 8
                    self.canvas.move(self.objekt, 8, 0)
        elif self.smer == "L":
            self.canvas.itemconfig(self.objekt, image=self.hrac_sprite["L"][self.faza])
            if (self.x - 12) // 32 > 0:
                if self.mapa[self.y // 32][(self.x - 12) // 32].hodnota == "0" or \
                   ((self.mapa[self.y // 32][self.x // 32].hodnota == "B") and \
                    (self.mapa[self.y // 32][(self.x - 32) // 32].hodnota == "0")):
                    self.x -= 8
                    self.canvas.move(self.objekt, -8, 0)            
        self.faza = (self.faza + 1) % 4

    def zomri(self): # metoda zmeni sprite hraca
        self.canvas.itemconfig(self.objekt, image=self.hrac_sprite["X"][0])
        self.canvas.update()

        
class Hra:
    class Vrchol:
        canvas = None
        def __init__(self, hodnota, y, x):
            self.x, self.y = x, y
            self.hodnota = hodnota
            self.susedia = set()
            self.casovac = 0
            self.navstiveny = False
            self.vybuch = False
            self.ohrozeny = False
            self.vylepsenie_typ = 0
            self.vylepsenie_objekt = None
        
        def pridaj_hranu(self, vrchol):
            self.susedia.add(vrchol)

        def odstran_hranu(self, vrchol):
            self.susedia.remove(vrchol)

        def pridaj_vylepsenie(self):
            if random.randrange(8) == 0:
                if random.randrange(2) == 0: # polozi na policko vylepsenie, ktore po zobrati zvysi kapacitu bomb
                    self.vylepsenie_objekt = self.canvas.create_image(1 + (self.x * 32) + 16, 1 + (self.y * 32) + 16, image=Hra.vylepsenia_sprite["plus_bomba"])
                    self.vylepsenie_typ = 1
                else: # polozi na policko vylepsenie, ktore po zobrati zvysi dlzku vybuchu
                    self.vylepsenie_objekt = self.canvas.create_image(1 + (self.x * 32) + 16, 1 + (self.y * 32) + 16, image=Hra.vylepsenia_sprite["plus_vybuch"])
                    self.vylepsenie_typ = 2

        def vymaz_vylepsenie(self): # metoda odstrani z plochy vylepsenie
            self.canvas.delete(self.vylepsenie_objekt)
            self.vylepsenie_typ = 0
        
        def __repr__(self):
            return f"<({self.y}, {self.x}), {self.hodnota}, {self.ohrozeny}>"
        
            
    canvas = None
    policko_sprite = None
    vylepsenia_sprite = None
    tlacidlo_spat = None
    bomby = None
    policka_s_vybuchom = None
    mapa = None
    vybuchy = None
    mapa_grafika = None
    
    def __init__(self, subor, dvaja_hraci=True):
        self.subor = subor
        self.hraci = []
        self.skore = {1 : 0, 2 : 0}
        self.hra_zapnuta = True
        self.skore_hrac = {1 : None , 2 : None}
        self.skore_max = 0
        self.ovladanie_pohybu = {"hrac_1" : [], "hrac_2" : []}
        self.klaves_poloz_bombu = {"hrac_1" : None, "hrac_2" : None}
        self.stav_klavesov = {"hore1" : False, "dole1" : False, "vlavo1" : False, "vpravo1" : False,
                              "hore2" : False, "dole2" : False, "vlavo2" : False, "vpravo2" : False}
        self.povol_vstup = True
        self.posledny_smer = None # posledny smer hraca 2 ovladaneho pc
        self.zastaveny = 0
        self.cesta = []
        self.mapa = Hrac.mapa = Bomba.mapa = Vybuch.mapa = []
        self.vybuchy = Bomba.vybuchy = []
        self.policka_s_vybuchom = Bomba.policka_s_vybuchom = Vybuch.policka_s_vybuchom = []
        self.bomby = Bomba.bomby = {}
        
        self.policko_sprite["zem"] = random.choice([self.policko_sprite["trava"], self.policko_sprite["zemina"],
                                                    self.policko_sprite["trava"]])

        self.policko_sprite["stena"] = random.choice([self.policko_sprite["kamen"], self.policko_sprite["kov stena"],
                                                      self.policko_sprite["hlina"], self.policko_sprite["tehla"],
                                                      self.policko_sprite["kov"], self.policko_sprite["kamen"]])
        
        self.precitaj_natavenia(dvaja_hraci)   
        self.precitaj_subor()
        self.vykresli_mapu()
        self.vykresli_tabulku()       
        self.vypis_skore()

        self.canvas.bind_all('<KeyPress>', self.stlac_klaves)
        self.canvas.bind_all('<KeyRelease>', self.pusti_klaves)
                
    def vykresli_tabulku(self): # vykresli okienko so skore
        self.canvas.create_rectangle(0, 416, 480, 480, fill="RoyalBlue1", outline="navy")
        self.canvas.create_text(0 + 8, 416, text="Skóre:", fill="white", anchor="nw", font="aerial 16")
        self.canvas.create_text(0 + 100, 416, text="Hráč 1", fill="white", anchor="nw", font="aerial 16")
        self.canvas.create_text(0 + 190, 416, text="Hráč 2", fill="black", anchor="nw", font="aerial 16")
        self.canvas.create_window(450, 450, width=40, height=40, window=self.tlacidlo_spat)
        self.canvas.update()

    def vypis_skore(self): # vypise hodnoty skore
        if self.skore_hrac[1] and self.skore_hrac[2]:
            self.canvas.itemconfig(self.skore_hrac[1], text=str(self.skore[1]))
            self.canvas.itemconfig(self.skore_hrac[2], text=str(self.skore[2]))
        else:          
            self.skore_hrac[1] = self.canvas.create_text(0 + 125, 448, text=str(self.skore[1]),
                              fill="white", anchor="nw", font="aerial 16")
            self.skore_hrac[2] = self.canvas.create_text(0 + 215, 448, text=str(self.skore[2]),
                              fill="black", anchor="nw", font="aerial 16")

    def nastavenie_vysky_sprite(self): #metoda urcuje ktory z hracov je perspektivne v zadu
        self.canvas.tag_raise(self.hraci[0].objekt)
        self.canvas.tag_raise(self.hraci[1].objekt)
        if self.hraci[1].y > self.hraci[0].y:
            self.canvas.lift(self.hraci[1].objekt)
        else:
            self.canvas.lift(self.hraci[0].objekt)
    
    def precitaj_subor(self, nahodna=True): # metoda precita subor a ak je nahodna True tak mapa ma nahodne umiestnene skatule ak nie tak ich umiestenie je precitane zo suboru               
        with open(self.subor) as subor:
            self.sur1 = tuple(map(int, subor.readline().split()))
            self.sur2 = tuple(map(int, subor.readline().split()))
            vyber = []
            for i, riadok in enumerate(subor):
                rad = []
                for j, prvok in enumerate(riadok.split()):
                    if nahodna and prvok in ("b", "0") and ((i, j) != self.sur1 and (i, j) != self.sur2):
                        rad.append(self.Vrchol("0", i, j))
                        vyber.append((i, j))
                    else:
                        prvok = "0" if prvok == "s" else prvok
                        rad.append(self.Vrchol(prvok, i, j))                  
                self.mapa.append(rad)
                
        if nahodna:
            policka_so_skatulou = random.choices(vyber, k=80) # pri teste bez skatul k=0, inak k=80
            for i, j in policka_so_skatulou:
                self.mapa[i][j] = self.Vrchol("b", i, j)
        pocet_riadkov, pocet_stlpcov = len(self.mapa),len(self.mapa[1])    
        for i in range(pocet_riadkov):
            for j in range(pocet_stlpcov):
                if i - 1 > 0:          
                    if self.mapa[i - 1][j].hodnota == "0" and self.mapa[i][j].hodnota == "0":
                        self.mapa[i][j].pridaj_hranu(self.mapa[i - 1][j])
                        self.mapa[i - 1][j].pridaj_hranu(self.mapa[i][j])
                
                if i + 1 < pocet_riadkov:
                    if self.mapa[i + 1][j].hodnota == "0" and self.mapa[i][j].hodnota == "0":
                        self.mapa[i][j].pridaj_hranu(self.mapa[i + 1][j])
                        self.mapa[i + 1][j].pridaj_hranu(self.mapa[i][j])
                        
                if j - 1 > 0:
                    if self.mapa[i][j - 1].hodnota == "0" and self.mapa[i][j].hodnota == "0":
                        self.mapa[i][j].pridaj_hranu(self.mapa[i][j - 1])
                        self.mapa[i][j - 1].pridaj_hranu(self.mapa[i][j])
                        
                if j + 1 < pocet_stlpcov:
                    if self.mapa[i][j + 1].hodnota == "0" and self.mapa[i][j].hodnota == "0":
                        self.mapa[i][j].pridaj_hranu(self.mapa[i][j + 1])
                        self.mapa[i][j + 1].pridaj_hranu(self.mapa[i][j])
                        
    def precitaj_natavenia(self, dvaja_hraci): # metoda precita subor, podla ktoreho rozlisi ktore klavesy ovladaju hracov 
        with open("nastavenia.txt") as nastavenia:
            self.skore_max = int(nastavenia.readline().strip())
            for i in range(4):
                self.ovladanie_pohybu["hrac_1"].append(nastavenia.readline().strip())
            self.klaves_poloz_bombu["hrac_1"] = nastavenia.readline().strip()
            if dvaja_hraci:
                for i in range(4):
                    self.ovladanie_pohybu["hrac_2"].append(nastavenia.readline().strip())
                self.klaves_poloz_bombu["hrac_2"] = nastavenia.readline().strip()
        
    def vykresli_mapu(self): # metoda vykresli mapu levela podla 2d-pola self.mapa
        self.skatule = {}
        self.hraci = []
        self.mapa_grafika = Vybuch.mapa_grafika = []
        if self.mapa:
            for i in range(len(self.mapa)):
                for j in range(len(self.mapa[i])):
                    if self.mapa[i][j].hodnota == "x":
                        self.mapa_grafika.append(self.canvas.create_image(1 + 16 + (j * 32), 1 + 16 + (i * 32),
                                                                          image=self.policko_sprite["stena"]))
                    elif self.mapa[i][j].hodnota == "b":
                        self.canvas.create_image(1 + 16 + (j * 32), 1 + 16 + (i * 32),
                                                 image=self.policko_sprite["zem"])
                        Vybuch.id_skatul[(j, i)] = self.skatule[(j, i)] = \
                                                   self.canvas.create_image(1 + 16 + (j * 32), 1 + 16 + (i * 32),
                                                                            image=self.policko_sprite["skatula"])                       
                    elif self.mapa[i][j].hodnota == "0":
                        self.mapa_grafika.append(self.canvas.create_image(1 + 16 + (j * 32), 1 + 16 + (i * 32),
                                                                          image=self.policko_sprite["zem"]))
                                            
        self.hraci.append(Hrac(*self.sur1))
        self.hraci.append(Hrac(*self.sur2, False))
        self.canvas.update()

    def vyherna_obrazovka(self, hrac): # metoda vypise vytaza hry
        self.canvas.delete("all")
        self.canvas.create_text(240, 200, text="Koniec hry!", fill="red", font="aerial 30")
        self.canvas.create_text(240, 245, text=f"Vyhral hráč {hrac}!", fill="red", font="aerial 20")
        self.canvas.update()
        
    def stlac_klaves(self, event): # vstup hraca, event riesi problem viacerych stlacenych klavesov v jednom momente
        klaves = event.keysym
        if klaves in self.ovladanie_pohybu["hrac_1"]:
            self.stav_klavesov[{0 : "hore1", 1 : "dole1", 2 : "vpravo1", 3 : "vlavo1"}
                               [self.ovladanie_pohybu["hrac_1"].index(klaves)]] = True # zisti index klavesu podla ktoreho stanovi smer
        elif klaves in self.ovladanie_pohybu["hrac_2"]:
            self.stav_klavesov[{0 : "hore2", 1 : "dole2", 2 : "vpravo2", 3 : "vlavo2"}
                               [self.ovladanie_pohybu["hrac_2"].index(klaves)]] = True # zisti index klavesu podla ktoreho stanovi smer
        elif klaves == self.klaves_poloz_bombu["hrac_1"]:
            x, y = (self.hraci[0].x // 32), (self.hraci[0].y // 32)
            if self.mapa[y][x].hodnota == "0" and self.povol_vstup \
               and self.hraci[0].pocet_bomb + 1 <= self.hraci[0].max_pocet_bomb: # event sa moze volat aj v momente ked ina funkcia iteruje slovnik, bez self.povol_vstup moze zriedka zhodit program
                self.bomby[(x, y)] = (Bomba(x, y, 1, self.hraci[0].dlzka_vybuchu))
                self.hraci[0].pocet_bomb += 1
        elif klaves == self.klaves_poloz_bombu["hrac_2"]:
            x, y = (self.hraci[1].x // 32), (self.hraci[1].y // 32)
            if self.mapa[y][x].hodnota == "0" and self.povol_vstup \
               and self.hraci[1].pocet_bomb + 1 <= self.hraci[1].max_pocet_bomb:     
                self.bomby[(x, y)] = (Bomba(x, y, 2, self.hraci[1].dlzka_vybuchu))
                self.hraci[1].pocet_bomb += 1

    def pusti_klaves(self, event): # vstup hraca, event riesi problem viacerych stlacenych klavesov v jednom momente
        klaves = event.keysym
        if klaves in self.ovladanie_pohybu["hrac_1"]:
            self.stav_klavesov[{0 : "hore1", 1 : "dole1", 2 : "vpravo1", 3 : "vlavo1"}
                               [self.ovladanie_pohybu["hrac_1"].index(klaves)]] = False
        elif klaves in self.ovladanie_pohybu["hrac_2"]:
            self.stav_klavesov[{0 : "hore2", 1 : "dole2", 2 : "vpravo2", 3 : "vlavo2"}
                               [self.ovladanie_pohybu["hrac_2"].index(klaves)]] = False

    def kontrola_vstupu(self, dvaja_hraci=True): # metoda kontroluje ktora pohybova klavesa bola stlacena a podrzana a zaroven hybe hracmi
        for smer, stav  in self.stav_klavesov.items():
            if stav and smer[-1] == "1":
                self.hraci[0].zmen_smer(smer[:-1])
            elif stav and smer[-1] == "2":
                self.hraci[1].zmen_smer(smer[:-1])

        if not self.stav_klavesov["hore1"] and not self.stav_klavesov["dole1"] \
        and not self.stav_klavesov["vlavo1"] and not self.stav_klavesov["vpravo1"]:
            self.hraci[0].zastav()
        if not self.stav_klavesov["hore2"] and not self.stav_klavesov["dole2"] \
        and not self.stav_klavesov["vlavo2"] and not self.stav_klavesov["vpravo2"] \
        and dvaja_hraci:
            self.hraci[1].zastav()

    def start(self, dvaja_hraci=True): # samotny beh hry
        self.restart = False
        self.n = 0
        hrac = self.hraci[0]
        while self.hra_zapnuta:
            zac = time.time()
            self.canvas.update()
            self.kontrola_vstupu(dvaja_hraci)
            if not dvaja_hraci:
                self.ovladaj_hraca_2()
            self.nastavenie_vysky_sprite()
            for hrac in self.hraci:
                hrac.animuj()
            kon = time.time()
            cas = 40 - (kon - zac) * 1000 if (kon - zac) * 1000 < 40 else 0 # cas sa stanovy podla rychlosti cyklu ak by cyklus zbehol za 0 ms tak pocka 40 ms 
            self.spracuj_bomby_a_vybuchy()
            self.spracuj_vylepsienia()
            self.canvas.after(int(cas))
                
    def restart_hry(self): # vymaze mapu, hracov, bomby... a vsetko vykresli odznovu
        self.mapa = Hrac.mapa = Bomba.mapa = Vybuch.mapa = []
        self.precitaj_subor()
        self.stav_klavesov = {"hore1" : False, "dole1" : False, "vlavo1" : False, "vpravo1" : False,
                              "hore2" : False, "dole2" : False, "vlavo2" : False, "vpravo2" : False}
        
        for idnt in self.mapa_grafika:          
            self.canvas.delete(idnt)

        for skatula in self.skatule.values():
            self.canvas.delete(skatula)

        for hrac in self.hraci:
            self.canvas.delete(hrac.objekt)
            
        self.hraci = []
        self.cesta = []
        self.vykresli_mapu()
        self.bomby = Bomba.bomby = {}
        self.vypis_skore()
        self.vybuchy = Bomba.vybuchy = []
        self.policka_s_vybuchom = Bomba.policka_s_vybuchom = Vybuch.policka_s_vybuchom = []
        self.canvas.bind_all('<KeyPress>', self.stlac_klaves)
        self.canvas.bind_all('<KeyRelease>', self.pusti_klaves)

    def spracuj_vylepsienia(self): # skontroluje ci nejake vylepsenie bolo zdvihnute a ktory hrac ho zdvihol
        if self.mapa[self.hraci[0].y // 32][self.hraci[0].x // 32].vylepsenie_typ == 1: # stoji v policku s vylepsenim max poctu polozenych bomb
            self.mapa[self.hraci[0].y // 32][self.hraci[0].x // 32].vymaz_vylepsenie()
            self.hraci[0].max_pocet_bomb += 1
            
        elif self.mapa[self.hraci[0].y // 32][self.hraci[0].x // 32].vylepsenie_typ == 2: # stoji v policku s vylepsenim dlzky vybuchu
            self.mapa[self.hraci[0].y // 32][self.hraci[0].x // 32].vymaz_vylepsenie()
            self.hraci[0].dlzka_vybuchu += 1

        if self.mapa[self.hraci[1].y // 32][self.hraci[1].x // 32].vylepsenie_typ == 1: # stoji v policku s vylepsenim max poctu polozenych bomb
            self.mapa[self.hraci[1].y // 32][self.hraci[1].x // 32].vymaz_vylepsenie()
            self.hraci[1].max_pocet_bomb += 1
            
        elif self.mapa[self.hraci[1].y // 32][self.hraci[1].x // 32].vylepsenie_typ == 2: # stoji v policku s vylepsenim dlzky vybuchu
            self.mapa[self.hraci[1].y // 32][self.hraci[1].x // 32].vymaz_vylepsenie()
            self.hraci[1].dlzka_vybuchu += 1
    
    def spracuj_bomby_a_vybuchy(self): # metoda animuje jednotlive instancie triedy Bomba a ich instancie Vybuchy, po vybuchu ich odstrani z pamati
        kos_vybuch = set()
        kos_bomba = set()
        self.povol_vstup = False
        for sur, bomba in self.bomby.items():
            if bomba.vybuchni:
                kos_bomba.add(sur)
                if bomba.majitel == 1:
                    self.hraci[0].pocet_bomb -= 1
                else:
                    self.hraci[1].pocet_bomb -= 1
            else:
                bomba.animuj()
        for vybuch in self.vybuchy:
            if vybuch.koniec_animacie:
                kos_vybuch.add(vybuch)
            else:
                vybuch.animuj()
        
        for vybuch in kos_vybuch:
            self.vybuchy.remove(vybuch)
        for sur in kos_bomba:
            del self.bomby[sur]

        self.kolizia_s_vybuchom()
        self.povol_vstup = True 

    def kolizia_s_vybuchom(self): # metoda kontroluje ci nastala kolizia hraca s vybuchom bomby, a pridava skore
        vybuchnuti = set()
        for i, hrac in enumerate(self.hraci):
            if self.mapa[hrac.y // 32][hrac.x // 32].vybuch:
                vybuchnuti.add(i)
                hrac.zomri()
                    
        if len(vybuchnuti) == 2:
            self.canvas.unbind(self.stlac_klaves)
            self.canvas.unbind(self.pusti_klaves)
            self.canvas.after(2000)
            self.restart_hry()
            
        elif len(vybuchnuti) == 1:
            self.canvas.unbind(self.stlac_klaves)
            self.canvas.unbind(self.pusti_klaves)
            
            if 1 in vybuchnuti: # 1 je v tomto priade oznacenie hraca 2
                self.skore[1] += 1
            else:
                self.skore[2] += 1
            
            if self.skore[1] == self.skore_max:
                self.vyherna_obrazovka("1")
                self.hra_zapnuta = False
                self.canvas.after(2000) 
            elif self.skore[2] == self.skore_max:
                self.vyherna_obrazovka("2")
                self.hra_zapnuta = False
                self.canvas.after(2000) 
            else:
                self.canvas.after(2000) 
                self.restart_hry()                     

    def ovladaj_hraca_2(self): # metoda sa v kazdom volani snazi posunut protihraca podla nejakych pravidiel
        self.naj_cesta = []
        nepriatel = self.hraci[1]
        hrac = self.hraci[0]
        
        def stanov_smer(x, y, ciel): # predpoklada ze ciel je vrchol susedny s hracom
            x1 = x // 32
            y1 = y // 32
            x2 = ciel.x
            y2 = ciel.y

            if y2 < y1:
                return "U"
            elif y2 > y1:
                return "D"
            elif x2 < x1:
                return "L"
            elif x2 > x1:
                return "R"
            else:
                return None
            
        def najdi_neohrozenu_cestu(cesta): # funkcia najde najblizsiu neohrozenu cestu backtrackingom
            vrchol = cesta[-1]
            
            if not vrchol.ohrozeny and self.naj_cesta == []:
                self.naj_cesta = cesta
            elif self.naj_cesta != [] and not vrchol.ohrozeny and len(cesta) < len(self.naj_cesta):
                self.naj_cesta = cesta
            elif self.naj_cesta != [] and len(cesta) >= len(self.naj_cesta):
                return
            else:
                for vrchol_2 in vrchol.susedia:
                    if not vrchol_2.navstiveny and not vrchol.vybuch and vrchol.casovac < 20:
                        vrchol_2.navstiveny = True                   
                        najdi_neohrozenu_cestu(cesta + [vrchol_2])
                        vrchol_2.navstiveny = False

        def prenasleduj_hraca(vrchol): # funkcia vrati najblizsiu cestu k hracovi
            hrac = self.hraci[0]
            navstivene = set()
            queue = [(vrchol, None)]
            
            while queue:
                vrchol, predchodca = queue.pop(0)
                if vrchol not in navstivene:
                    navstivene.add(vrchol)
                    vrchol.pred = predchodca
                    if vrchol == self.mapa[hrac.y // 32][hrac.x // 32]:
                        cesta = []
                        while vrchol:
                            cesta.append(vrchol)
                            vrchol = vrchol.pred                           
                        return cesta[::-1]
                    for vrchol_2 in vrchol.susedia:
                        if vrchol_2 not in navstivene:
                            queue.append((vrchol_2, vrchol))
            return []
        
        def najdi_vylepsenie(vrchol): # funkcia vrati najblizsiu cestu k vylepseniu
            navstivene = set()
            queue = [(vrchol, None)]
            while queue:
                vrchol, predchodca = queue.pop(0)
                if vrchol not in navstivene:
                    navstivene.add(vrchol)
                    vrchol.pred = predchodca
                    if vrchol.vylepsenie_typ:
                        cesta = []
                        while vrchol:
                            cesta.append(vrchol)
                            vrchol = vrchol.pred                           
                        return cesta[::-1]
                    for vrchol_2 in vrchol.susedia:
                        if vrchol_2 not in navstivene:
                            queue.append((vrchol_2, vrchol))
            return []

        def najdi_skatulu(vrchol): # funkcia najde cestu ku skatuli
            def sused_skatula(vrchol): # funkcia zisti ci je vrchol potencialnym susedom
                x, y = vrchol.x, vrchol.y
                if x - 1 >= 0:
                    if self.mapa[y][x - 1].hodnota == "b":
                        return True
                if x + 1 < len(self.mapa[y]):
                    if self.mapa[y][x + 1].hodnota == "b":
                        return True
                if y - 1 >= 0:
                    if self.mapa[y - 1][x].hodnota == "b":
                        return True
                if y + 1 < len(self.mapa):
                    if self.mapa[y + 1][x].hodnota == "b":
                        return True
                return False
                    
            navstivene = set()
            queue = [(vrchol, None)]
            while queue:
                vrchol, predchodca = queue.pop(0)
                if vrchol not in navstivene:
                    navstivene.add(vrchol)
                    vrchol.pred = predchodca
                    if sused_skatula(vrchol):
                        cesta = []
                        while vrchol:
                            cesta.append(vrchol)
                            vrchol = vrchol.pred                           
                        return cesta[::-1]
                    for vrchol_2 in vrchol.susedia:
                        if vrchol_2 not in navstivene and not vrchol.ohrozeny:
                            queue.append((vrchol_2, vrchol))
            return []
                        
        def mozem_polozit_bombu(): # funkcia zisti ci ma nepriatel moznost najst ukryt ak by polozil bombu
            x, y = nepriatel.x // 32, nepriatel.y // 32
            prekazka = {"L" : False, "R" : False, "U" : False, "D" : False}
            
            if self.hraci[1].pocet_bomb + 1 > self.hraci[1].max_pocet_bomb or self.mapa[y][x].hodnota == "B":
                return False
            if not self.mapa[y][x].ohrozeny:
                vrcholy = [self.mapa[y][x]]
                self.mapa[y][x].ohrozeny = True
            else:
                vrcholy = []
            
            for i in range(nepriatel.dlzka_vybuchu + 1):
                if x + 1 + i < len(self.mapa[y]):
                    if self.mapa[y][x + 1 + i].hodnota == "0" and not prekazka["D"] and \
                       not self.mapa[y][x + 1 + i].ohrozeny:
                        vrcholy.append(self.mapa[y][x + 1 + i])
                        self.mapa[y][x + 1 + i].ohrozeny = True
                    elif self.mapa[y][x + 1 + i].hodnota == "b":
                        prekazka["D"] = True
                if x - 1 - i >= 0:
                    if self.mapa[y][x - 1 - i].hodnota == "0" and not prekazka["U"] and \
                       not self.mapa[y][x - 1 - i].ohrozeny:
                        vrcholy.append(self.mapa[y][x - 1 - i])
                        self.mapa[y][x - 1 - i].ohrozeny = True
                    elif self.mapa[y][x - 1 - i].hodnota == "b":
                        prekazka["U"] = True
                if y + 1 + i < len(self.mapa):
                    if self.mapa[y + 1 + i][x].hodnota == "0" and not prekazka["R"] and \
                       not self.mapa[y + 1 + i][x].ohrozeny:
                        vrcholy.append(self.mapa[y + 1 + i][x])
                        self.mapa[y + 1 + i][x].ohrozeny = True
                    elif self.mapa[y + 1 + i][x].hodnota == "b":
                        prekazka["R"] = True
                if y - 1 - i >= 0:
                    if self.mapa[y - 1 - i][x].hodnota == "0" and not prekazka["L"] and \
                       not self.mapa[y - 1 - i][x].ohrozeny:
                        vrcholy.append(self.mapa[y - 1 - i][x])
                        self.mapa[y - 1 - i][x].ohrozeny = True
                    elif self.mapa[y - 1 - i][x].hodnota == "b":
                        prekazka["L"] = True
                        
            najdi_neohrozenu_cestu([self.mapa[y][x]])
            policko = self.naj_cesta[1] if len(self.naj_cesta) > 1 else None
            for vrchol in vrcholy:
                vrchol.ohrozeny = False
            if policko:
                return True
            return False

        def som_v_blizkosti_protihraca(): # funkcia zistuje ci by polozena bomba mohla zasiahnut hraca
            x1, y1 = nepriatel.x // 32, nepriatel.y // 32
            x2, y2 = hrac.x // 32, hrac.y // 32
            if abs(x1 - x2) <= (nepriatel.dlzka_vybuchu + 1) and y1 == y2:
                return True
            elif abs(y1 - y2) <= (nepriatel.dlzka_vybuchu + 1) and x1 == x2:
                return True
            return False

        def som_v_blizkosti_skatule(): # funkcia zistuje ci je nepriatel blizko skatule
            x, y = nepriatel.x // 32, nepriatel.y // 32
            prekazka = {"L" : False, "R" : False, "U" : False, "D" : False}
            for i in range(nepriatel.dlzka_vybuchu + 2):
                if x + i < len(self.mapa[y]) and not prekazka["R"]:
                    if self.mapa[y][x + i].hodnota == "b":
                        return True
                    elif self.mapa[y][x + i].hodnota == "x":
                        prekazka["R"] = True
                if x - i >= 0 and not prekazka["L"]:
                    if self.mapa[y][x - i].hodnota == "b":
                        return True
                    elif self.mapa[y][x - i].hodnota == "x":
                        prekazka["L"] = True
                if y + i < len(self.mapa) and not prekazka["D"]:
                    if self.mapa[y + i][x].hodnota == "b":
                        return True
                    elif self.mapa[y + i][x].hodnota == "x":
                        prekazka["D"] = True                  
                if y - i >= 0 and not prekazka["U"]:
                    if self.mapa[y - i][x].hodnota == "b":
                        return True
                    elif self.mapa[y - i][x].hodnota == "x":
                        prekazka["U"] = True
            return False

        def vyber_si_nahodnu_cestu(cesta): # funkcia vyberie nahodny neohrozeny vrchol v blizosti nepriatela
            vrchol = cesta[-1]
            
            if not vrchol.ohrozeny and random.randrange(3) == 0:
                self.naj_cesta = cesta
                print(cesta)
                return
            
            else:
                for vrchol_2 in vrchol.susedia:
                    if not vrchol_2.navstiveny and not vrchol.vybuch and vrchol.casovac < 20:
                        vrchol_2.navstiveny = True                   
                        najdi_neohrozenu_cestu(cesta + [vrchol_2])
                        vrchol_2.navstiveny = False       

        mozem_polozit_bombu = mozem_polozit_bombu()
        policko = None  
        if len(self.cesta) > 0: # ak mam nejaku cestu
            if self.mapa[(nepriatel.y) // 32][(nepriatel.x) // 32] == self.cesta[0]:
                self.cesta.pop(0)
                policko = self.cesta[0] if len(self.cesta) > 0 else None
            else:
                policko = self.cesta[0] if not self.cesta[0].ohrozeny else None
                
        if (som_v_blizkosti_protihraca() or som_v_blizkosti_skatule()) and mozem_polozit_bombu: # som blizko hraca alebo skatule a mam sa kam schovat?
            self.bomby[(nepriatel.x // 32, nepriatel.y // 32)] = Bomba(nepriatel.x // 32, nepriatel.y // 32, 2, nepriatel.dlzka_vybuchu)
            nepriatel.pocet_bomb += 1

        if self.mapa[nepriatel.y // 32][nepriatel.x // 32].ohrozeny: # ak mu hrozi nebezpecie tak sa schova
            najdi_neohrozenu_cestu([self.mapa[nepriatel.y // 32][nepriatel.x // 32]])
            if len(self.naj_cesta) > 1:
                policko = self.naj_cesta[1]
                self.cesta = self.naj_cesta
            
        if not policko: # hladaj hraca
            cesta_k_hracovi = prenasleduj_hraca(self.mapa[nepriatel.y // 32][nepriatel.x // 32])        
            if len(cesta_k_hracovi) > 1 and random.randrange(2) == 0:
                self.cesta = cesta_k_hracovi[1:]
                policko = self.cesta[0] if self.cesta[0].casovac < 15 else None

        if not policko: # zober vylepsenie
            cesta_k_vylepseniu = najdi_vylepsenie(self.mapa[nepriatel.y // 32][nepriatel.x // 32])
            if len(cesta_k_vylepseniu) > 1 and random.randrange(3) == 0:
                self.cesta = cesta_k_vylepseniu[1:]
                policko = cesta_k_vylepseniu[1] if not cesta_k_vylepseniu[1].ohrozeny else None

        if not policko: # najdi skatulu
            cesta_ku_skatuli = najdi_skatulu(self.mapa[nepriatel.y // 32][nepriatel.x // 32])
            if len(cesta_ku_skatuli) > 1 and random.randrange(2) == 0:
                self.cesta = cesta_ku_skatuli[1:]
                policko = cesta_ku_skatuli[1] if cesta_ku_skatuli[1].casovac < 15 else None
        
        if policko is None and self.zastaveny > 40: # zasekol sa     
            vyber_si_nahodnu_cestu([self.mapa[(nepriatel.y) // 32][(nepriatel.x) // 32]])
            if len(self.naj_cesta) > 1:
                self.zastaveny = 0
                policko = self.naj_cesta[1] # vyber sa k nejakemu nahodneu policku
                self.cesta = self.naj_cesta[1:]
            
        if policko: # vrchol ku ktoremu sa mam vydat
            self.zastaveny = 0
            smer = stanov_smer(nepriatel.x, nepriatel.y, policko) # smer akym sa mam k tomu vrcholu dostat
            self.posledny_smer = smer
            nepriatel.faza = nepriatel.faza if nepriatel.smer == smer else 0 # v pripade ze zmeni smer animacia zacne of fazy 0
            nepriatel.smer = smer
        else: # ak nemam co robit tak zastane nepriatela a zobrazi ho spravnym spriteom
            nepriatel.zastav()
            self.zastaveny += 1

       
hra = Menu()
