from PIL import Image
import tkinter
import random

#???level_8 nefunguje, konkretne ked viacero mys naraz vlezie do jedneho domceka v jednom momente???
#obrazok jamy prekryva steny a je teraz zle centrovany
# pohyb mysou zastavuje hru v leveli ked je spristupnena zmena farby tlacidla X

class Hra:
    canvas = None
    obrazok_pozadia = None
    logo = None
    anim_chuchu = None
    anim_kapukapu = None
    objekty = None
    
    def __init__(self):
        self.hlavne_menu = False
        self.vyber = False
        self.pomoc = False
        self.spusteny_level = False        
        self.zvoleny = None
        self.level = None
        self.policka = []
        self.nazov_levelu = None
        self.spat = None
        self.chuchu = []
        self.kapukapu = []

        hra_okno = tkinter.Tk()
        hra_okno.title("ChuChu Rocket!")
        
        self.canvas = Level.canvas = PohybujuciObjekt.canvas = Sipka.canvas = Dom.canvas = tkinter.Canvas(width=568, height=384, background="black")
        self.canvas.pack()
        
        
        self.obrazok_pozadia = tkinter.PhotoImage(file="grafika/pozadie.png")        
        self.logo = tkinter.PhotoImage(file="grafika/logo.png")
        self.anim_chuchu = [tkinter.PhotoImage(file=f"ChuChu/chuchu{i}.png") for i in range(10)]
        self.anim_kapukapu = [tkinter.PhotoImage(file=f"KapuKapu/kapukapu{i}.png") for i in range(16, 32)]
        self.objekty = [tkinter.PhotoImage(file=f"objekty/panel{i}.png") for i in range(1, 8, 2)]
        [self.objekty.append(tkinter.PhotoImage(file=f"objekty/{nazov}.png")) for nazov in ("dom1", "jama")]
        self.vykresli_menu()
        
        self.canvas.bind("<Motion>", self.pohyb)
        self.canvas.bind("<ButtonPress-1>", self.klik)
        self.canvas.bind('<B1-Motion>', self.tahanie)
        self.canvas.bind('<ButtonRelease-1>', self.pustenie)
        
        tkinter.mainloop()
        
    def vykresli_menu(self):        
        self.hlavne_menu = True
        self.vyber = False
        self.pomoc = False
        self.spusteny_level = False
        self.canvas.delete("all")
        self.policka = []
        
        self.pozadie = self.canvas.create_image(285, 193, image=self.obrazok_pozadia)
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = 20, 192, 20, 242, 270, 242, 285, 227, 285, 207, 270, 192
        self.polozka_1 = self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, fill="navy")
        self.canvas.create_text(x1 + 125, y1 + 25, text="Puzzle", fill="white", font="Verdana 22")
        self.polozka_2 = self.canvas.create_polygon(x1, y1 + 64, x2, y2 + 64, x3, y3 + 64, x4, y4 + 64, x5, y5 + 64, x6, y6 + 64, fill="navy")
        self.canvas.create_text(x1 + 125, y1 + 89, text="Pomoc", fill="white", font="Verdana 22")
        self.polozka_3 = self.canvas.create_polygon(x1, y1 + 128, x2, y2 + 128, x3, y3 + 128, x4, y4 + 128, x5, y5 + 128, x6, y6 + 128, fill="navy")
        self.canvas.create_text(x1 + 125, y1 + 153, text="???", fill="white", font="Verdana 22")

    def vykresli_vyber_urovne(self):
        self.hlavne_menu = False
        self.vyber = True
        self.pomoc = False
        self.spusteny_level = False
        self.canvas.delete("all")
        self.policka = []
        
        self.canvas.create_image(130, 83, image=self.logo) #logo
        self.canvas.create_rectangle(42, 177, 226, 267, fill="white", outline="navy", width=5) #okno pre nazov
        self.canvas.create_text(134, 197, text="Meno levelu:", font="Verdana 16") #nazov
        self.nazov_levelu = self.canvas.create_text(134, 242, text="", font="Verdana 12") #meno        
        self.canvas.create_rectangle(264, 42, 530, 82, fill="white", outline="navy", width=6) #okno pre titulok
        self.spat = self.canvas.create_rectangle(486, 45, 528, 82, fill="red", outline="") #tlacidlo spat
        self.canvas.create_text(506, 62, text="X", fill="white", font="Verdana 16")
        self.canvas.create_text(380, 60, text="Puzzle", font="Verdana 22") #titulok
        self.canvas.create_rectangle(266, 82, 528, 344, fill="white", outline="navy", width=10) #okno pre okienka
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8 = 276, 102, 276, 122, 286, 132, 306, 132, 316, 122, 316, 102, 306, 92, 286, 92 #suradnice polygonu
        with open("progres.txt") as subor:
            progres = subor.readline().split()
        prejdene = [int(prvok) for prvok in progres]
        for i in range(5): #vykreslenie okienok
            for j in range(5):
                self.policka.append(self.canvas.create_polygon(x1 + (j * 50), y1 + (i * 50), x2 + (j * 50), y2 + (i * 50), x3 + (j * 50), y3 + (i * 50), x4 + (j * 50), y4 + (i * 50), x5 + (j * 50), y5 + (i * 50), x6 + (j * 50), y6 + (i * 50), x7 + (j * 50), y7 + (i * 50), x8 + (j * 50), y8 + (i * 50), fill="navy"))
                self.canvas.create_text(296 + j * 50, 112 + i * 50, text=f"{i * 5 + j + 1}", fill="white", font="Verdana 16")
                if i * 5 + j in prejdene:
                    self.canvas.create_text(296 + j * 50, 112 + i * 50, text="X", fill="green", font="Verdana 30")

    def vykresli_pomoc(self):
        self.hlavne_menu = False
        self.vyber = False
        self.pomoc = True
        self.spusteny_level = False
        self.canvas.delete("all")
        self.policka = []
        
        self.canvas.create_rectangle(42, 82, 526, 342, fill="gold2", outline="orange", width=5) #okno s instrukciami
        self.canvas.create_rectangle(42, 42, 526, 82, fill="white", outline="navy", width=5) #okno pre titulok
        self.canvas.create_text(264, 62, text="Pomoc", fill="black", font="Verdana 16")#titulok
        self.spat = self.canvas.create_rectangle(489, 45, 524, 80, fill="red", outline="") #tlacidlo spat
        self.canvas.create_text(506, 62, text="X", fill="white", font="Verdana 16")
        self.canvas.create_text(92, 92, text="Tvojou úlohou je dostať všetky myšky ChuChu do modrého \ndomčeka. V prípade, že myš narazí na stenu otočí sa vpravo.", anchor="nw", font="Verdana 10")
        self.chuchu = self.canvas.create_image(68, 108, image=self.anim_chuchu[0])
        self.canvas.create_text(92, 134, text="Každá úroveň ma daný počet a typ panelov, ktoré slúžia \nna usmernenie pohybu myší/mačiek. Kým je hra pozastavená \nmôžes tieto panely umiestnovať ťahaním do plochy.", anchor="nw", font="Verdana 10")
        self.canvas.create_image(68, 160, image=self.objekty[0])
        self.canvas.create_text(92, 190, text="Počas hry nesmie ani jednu myš zjesť mačka KapuKapu. Taktiež \nani jedna mačka nesmie vstúpiť do domčeku s myškami, kým \nmyšky neodleteli", anchor="nw", font="Verdana 10")
        self.kapukapu = self.canvas.create_image(68, 212, image=self.anim_kapukapu[0])
        self.canvas.create_text(92, 264, text="Toto je jama a počas hry nesmie do nej spadnúť ani jedna myš.", anchor="nw", font="Verdana 10")
        self.canvas.create_image(68, 276, image=self.objekty[5])
        self.canvas.create_text(92, 302, text="Hru vyhrávaš, keď všetky myšky vojdu do domčeka, myšky sa \nsa evakuujú a odletia na rakete ", anchor="nw", font="Verdana 10")
        self.canvas.create_image(68, 318, image=self.objekty[4])
        
        self.animuj()
        
    def animuj(self):
        faza1 = 0
        faza2 = 0
        while self.pomoc:
            
            faza1 = (faza1 + 1) % len(self.anim_chuchu)
            faza2 = (faza2 + 1) % len(self.anim_kapukapu)
            
            self.canvas.itemconfig(self.chuchu, image=self.anim_chuchu[faza1])
            self.canvas.itemconfig(self.kapukapu, image=self.anim_kapukapu[faza2])
            
            self.canvas.update()
            self.canvas.after(100)

    def spusti_level(self):
        kos_maciek = []
        kos_mysi = []
        navstivene_domceky = []
        while self.spusteny_level:
            for dom in self.level.domceky:
                if dom.jama is False:
                    dom.reset()
            if self.level.start_hry and self.level.zastav_hru is False:
                kos_maciek = []
                kos_mysi = []
                    
                for objekt in self.level.mysi:
                    objekt.posun()
                    
                    if objekt.zomrel:                        
                        self.level.chyba(objekt.objekt)
                    if objekt.vymaz: #mys pravdepodobne vstupila do domceku                       
                        for dom in self.level.domceky: #kontrola domceku do ktoreho vosla
                            if objekt.stlpec == dom.stlpec and objekt.riadok == dom.riadok:
                                dom.vstup()
                                navstivene_domceky.append(dom)
                                break 
                        self.canvas.delete(objekt.objekt)
                        kos_mysi.append(objekt)
                        
                for objekt in self.level.macky:
                    objekt.posun()
                    
                    if objekt.vymaz is True: #macka pravdepodobne vstupila do domceku                       
                        for dom in self.level.domceky: #kontrola domceku do ktoreho vosla
                            if objekt.stlpec == dom.stlpec and objekt.riadok == dom.riadok:                                
                                if dom.jama is False: #v pripade ze z domu vystrelila raketa
                                    dom.vstup()
                                    navstivene_domceky.append(dom)                                    
                                    self.level.chyba(objekt.objekt)
                                break
                        self.canvas.delete(objekt.objekt)
                        kos_maciek.append(objekt)
                        
                    
                for objekt in kos_maciek:
                    self.level.macky.remove(objekt)
                    del objekt                    
                for objekt in kos_mysi:
                    self.level.mysi.remove(objekt)
                    del objekt

                self.level.kolizia_s_mackou()
                
                if len(self.level.mysi) == 0:                        
                    for dom in self.level.domceky:
                        dom.odpal()
                        if dom.faza_animacie == len(dom.raketa_anim):
                            with open("progres.txt") as subor:
                                progres = subor.readline().split()
                            progres.append(str(self.level.identifikator))
                            with open("progres.txt", "w") as subor:
                                print(" ".join(progres), file=subor)
                            self.spusteny_level = False

            else:
                if self.level.kruh is not None:
                    self.level.animuj_kruh()
                for objekt in self.level.mysi:
                    objekt.animuj()
                    if objekt.vymaz:
                        self.canvas.delete(objekt.objekt)                        
                for objekt in self.level.macky:
                    objekt.animuj()
            self.canvas.update()
            self.canvas.after(self.level.rychlost)
        self.vykresli_vyber_urovne()
            
    def pohyb(self, event):
        x, y = event.x, event.y

        if self.hlavne_menu: #spravanie funkcie ked je spustene menu
            if 20 <= x <= 285 and 192 <= y <= 370:
                b = (y - 192) // 64
                if b * (64) < y - 192 < b * (64) + 50:
                    vyber = {0 : self.polozka_1, 1 : self.polozka_2, 2: self.polozka_3}[b]
                    self.canvas.itemconfig(vyber, fill="gold2")
                    for polozka in [self.polozka_1, self.polozka_2, self.polozka_3]:
                        if polozka != vyber:
                            self.canvas.itemconfig(polozka, fill="navy")
                else:
                    for polozka in [self.polozka_1, self.polozka_2, self.polozka_3]:
                        self.canvas.itemconfig(polozka, fill="navy")
            else:
                for polozka in [self.polozka_1, self.polozka_2, self.polozka_3]:
                     self.canvas.itemconfig(polozka, fill="navy")

        elif self.vyber: #spravanie funkcie ked je spusteny vyber puzzle
            if (486 <= x <= 526) and (42 <= y <= 82):
                self.canvas.itemconfig(self.spat, fill="red4")
            else:
                self.canvas.itemconfig(self.spat, fill="red")
                
            if 276 <= x <= 526 and 92 <= y <= 342:
                b = (y - 92) // 50
                a = (x - 276) // 50
                if (b * (50) < y - 92 < b * (50) + 40) and (a * (50) < x - 276 < a * (50) + 40):
                    vyber = self.policka[5 * b + a]
                    self.canvas.itemconfig(vyber, fill="gold2")
                    try:
                        with open(f"levely/lahke/level_{b * 5 + a + 1}.txt", encoding='utf-8') as subor:
                            meno = subor.readline()[7:]
                        self.canvas.itemconfig(self.nazov_levelu, text=meno)
                    except:
                        self.canvas.itemconfig(self.nazov_levelu, text="???")
                        
                    for okienko in self.policka:
                        if okienko != vyber:
                            self.canvas.itemconfig(okienko, fill="navy")
                else:
                    for okienko in self.policka:
                        self.canvas.itemconfig(okienko, fill="navy")
                    self.canvas.itemconfig(self.nazov_levelu, text="")
            else:
                for okienko in self.policka:
                    self.canvas.itemconfig(okienko, fill="navy")
                self.canvas.itemconfig(self.nazov_levelu, text="")

        elif self.pomoc: #spravanie funkcie ked je spustena pomoc
            if (486 <= x <= 526) and (42 <= y <= 82):
                self.canvas.itemconfig(self.spat, fill="red4")
            else:
                self.canvas.itemconfig(self.spat, fill="red")

        elif self.spusteny_level: #spravanie funkcie ked je spusteny level
            if (32 <= x <= 120) and (32 <= y <= 56):
                self.canvas.itemconfig(self.level.start, fill="gold2")
            else:
                self.canvas.itemconfig(self.level.start, fill="white")
                
            if (32 <= x <= 120) and (64 <= y <= 88) and self.level.start_hry:
                self.canvas.itemconfig(self.level.restart, fill="gold2")
            else:
                self.canvas.itemconfig(self.level.restart, fill="white")
            
            """
            if (496 <= x <= 536) and (328 <= y <= 368) and (len(self.level.mysi) != 0):
                self.canvas.itemconfig(self.level.spat, fill="red4")
            else:
                self.canvas.itemconfig(self.level.spat, fill="red")
            """
                
                       
    def klik(self, event):
        x, y = event.x, event.y
                
        if self.hlavne_menu: #spravanie funkcie ked je spustene menu
            if 20 <= x <= 285 and 192 <= y <= 370:
                b = (y - 192) // 64
                if b * (64) < y - 192 < b * (64) + 50:
                    if b < 2:
                        {0 : self.vykresli_vyber_urovne, 1 : self.vykresli_pomoc, 2 : "???"}[b]()
                    
        elif self.vyber: #spravanie funkcie ked je spusteny vyber puzzle
            if (486 <= x <= 526) and (42 <= y <= 82):
                self.vykresli_menu()
            if 276 <= x <= 526 and 92 <= y <= 342:
                b = (y - 92) // 50
                a = (x - 276) // 50
                if (b * (50) < y - 92 < b * (50) + 40) and (a * (50) < x - 276 < a * (50) + 40):
                    self.hlavne_menu = False
                    self.vyber = False
                    self.pomoc = False
                    self.spusteny_level = True
                    self.canvas.delete("all")
                    self.policka = []

                    self.level = Level(f"levely/lahke/level_{5 * b + a + 1}.txt")
                    self.level.identifikator = 5 * b + a
                    self.spusti_level()
                    
        elif self.pomoc: #spravanie funkcie ked je spustena pomoc
            if (486 <= x <= 526) and (42 <= y <= 82):
                self.vykresli_menu()

        elif self.spusteny_level: #spravanie funkcie ked je spusteny level
            if (496 <= x <= 536) and (328 <= y <= 368) and (len(self.level.mysi) != 0):
                self.spusteny_level = False
                
            if (32 <= x <= 120) and (32 <= y <= 56):
                if self.level.start_hry:
                    if self.level.rychlost == 20:
                        self.level.rychlost = 50
                    else:
                        self.level.rychlost = 20
                else:
                    self.level.start_hry = True

            elif (32 <= x <= 120) and (64 <= y <= 88) and self.level.start_hry:
                self.canvas.itemconfig(self.level.restart, fill="white")
                self.level.restart_levelu()
            
            elif self.level.start_hry is False:    
                for sipka in (self.level.sipky_v_panely):
                    if (sipka.x - 16 < x < sipka.x + 16) and (sipka.y - 16 < y < sipka.y + 16):
                        sipka.kliknutie_na_sipku = True
                        if 152 < x < 536 and 40 < y < 328:
                            x1 = int((x - 152) / 32)
                            y1 = int((y - 40) / 32)
                            if PohybujuciObjekt.mapa_pohybujucich_objektov[1 + (y1 * 2)][1 + (x1 * 2)] == sipka.smer:
                                PohybujuciObjekt.mapa_pohybujucich_objektov[1 + (y1 * 2)][1 + (x1 * 2)] = "_"        
                            else:
                                sipka.reset()
                        else:
                            sipka.reset()

    def tahanie(self, event):
        x, y = event.x, event.y
        if self.spusteny_level: #spravanie funkcie ked je spusteny level
            if self.level.start_hry is False:
                for sipka in (self.level.sipky_v_panely):
                    if sipka.kliknutie_na_sipku == True:
                        self.canvas.tag_raise(sipka.stvorec)
                        self.canvas.move(sipka.stvorec, x - sipka.x, y - sipka.y)
                        sipka.x, sipka.y = x, y

    def pustenie(self, event):
        x, y = event.x, event.y
        if self.spusteny_level: #spravanie funkcie ked je spusteny level
            if self.level.start_hry is False:
                for sipka in reversed(self.level.sipky_v_panely):
                    if sipka.kliknutie_na_sipku == True:
                        if 152 < x < 536 and 40 < y < 328:
                            x1 = int((x - 152) / 32)
                            y1 = int((y - 40) / 32)
                            if PohybujuciObjekt.mapa_pohybujucich_objektov[1 + (y1 * 2)][1 + (x1 * 2)] == "_":
                                PohybujuciObjekt.mapa_pohybujucich_objektov[1 + (y1 * 2)][1 + (x1 * 2)] = sipka.smer
                                self.canvas.move(sipka.stvorec, -1 * sipka.x+ (152 + (x1 * 32) + 16 ), -1 * sipka.y + (40 + (y1 * 32) + 16))
                                sipka.x = 152 + (x1 * 32) + 16 
                                sipka.y = 40 + (y1 * 32) + 16

                            else:
                                sipka.reset()
                        else:
                            sipka.reset()
                        sipka.kliknutie_na_sipku = False     
        
class Level:
    canvas = None
    policka = None
    kruhy = None
    
    def __init__(self, meno_levelu):
        self.canvas.delete("all")
        self.identifikator = None
        self.zastav_hru = False
        self.start_hry = False
        self.zoznam_sipiek = []
        self.mapa_levelu = []
        self.mysi = []
        self.macky = []
        self.sipky_v_panely = []
        self.domceky = []
        self.kruh = None
        self.rychlost = 50
        self.faza_kruhu = 0

        self.start = self.canvas.create_rectangle(32, 32, 120, 56, fill="white",
                                                  outline="navy", width=6) #Start
        self.restart = self.canvas.create_rectangle(32, 64, 120, 88, fill="white",
                                                    outline="navy", width=6) #Cancel
        self.canvas.create_rectangle(152, 328, 536, 368, fill="white", outline="navy",
                                     width=6) #okno pre meno levelu
        self.spat = self.canvas.create_rectangle(496, 329, 534, 365,
                                                 fill="red", outline="") #tlacidlo spat   


        
        PohybujuciObjekt.chuchu_d = [tkinter.PhotoImage(file=f"ChuChu/chuchu{i}.png") for i in range(10)]
        PohybujuciObjekt.chuchu_r = [tkinter.PhotoImage(file=f"ChuChu/chuchu{i}.png") for i in range(10, 20)]
        PohybujuciObjekt.chuchu_l = [tkinter.PhotoImage(file=f"ChuChu/chuchu{i}.png") for i in range(20, 30)]
        PohybujuciObjekt.chuchu_u = [tkinter.PhotoImage(file=f"ChuChu/chuchu{i}.png") for i in range(30, 40)]
        PohybujuciObjekt.chuchu_anj = [tkinter.PhotoImage(file=f"ChuChu/chuchu{i}.png") for i in range(40, 48)]
        PohybujuciObjekt.chuchu_pad = [tkinter.PhotoImage(file=f"ChuChu/chuchu{i}.png") for i in range(56, 63)]
        PohybujuciObjekt.kapukapu_d = [tkinter.PhotoImage(file=f"KapuKapu/kapukapu{i}.png") for i in range(16)]
        PohybujuciObjekt.kapukapu_r = [tkinter.PhotoImage(file=f"KapuKapu/kapukapu{i}.png") for i in range(16, 32)]
        PohybujuciObjekt.kapukapu_l = [tkinter.PhotoImage(file=f"KapuKapu/kapukapu{i}.png") for i in range(32, 48)]
        PohybujuciObjekt.kapukapu_u = [tkinter.PhotoImage(file=f"KapuKapu/kapukapu{i}.png") for i in range(48, 64)]
        PohybujuciObjekt.kapukapu_pad = [tkinter.PhotoImage(file=f"KapuKapu/kapukapu{i}.png") for i in range(64, 77)]
        Sipka.sipky = [tkinter.PhotoImage(file=f"objekty/panel{i}.png") for i in range(8)]
        Dom.domcek = [tkinter.PhotoImage(file=f"objekty/dom{i}.png") for i in range(4)]
        Dom.rakety_UL = [tkinter.PhotoImage(file=f"rakety/raketa{i}.png") for i in range(60)]
        Dom.rakety_UR = [tkinter.PhotoImage(file=f"rakety/raketa{i}.png") for i in range(60, 120)]
        Dom.rakety_DL = [tkinter.PhotoImage(file=f"rakety/raketa{i}.png") for i in range(120, 180)]
        Dom.rakety_DR = [tkinter.PhotoImage(file=f"rakety/raketa{i}.png") for i in range(180, 240)]
        Dom.raketa_dym = [tkinter.PhotoImage(file=f"rakety/dym{i}.png") for i in range(60)]
        self.policka = [tkinter.PhotoImage(file="objekty/jama.png")]
        self.kruhy = [tkinter.PhotoImage(file=f"kruhy/kruh{i}.png") for i in range(0, 50, 2)]

        self.UI_levelu()
        self.precitaj_subor(meno_levelu)
        self.vykresli_level()
        
    def UI_levelu(self):        
        self.canvas.create_text(76, 44, text="Štart/Turbo", font="Verdana 10", fill="navy")
        self.canvas.create_text(76, 76, text = "Reštart", font="Verdana 10", fill="navy")
        self.canvas.create_rectangle(32, 96, 120, 344, fill="white",
                                     outline="navy", width=6) #Panel so sipkami
        self.canvas.create_rectangle(152, 40, 536, 328, fill="white",
                                     width=1, outline="red") #Herna plocha
     
        
        self.canvas.create_text(516, 348, text="X", fill="white", font="Verdana 16")
        farba1, farba2 = tuple(random.sample(("yellow", "pink", "SeaGreen1",
                                              "orange", "plum1", "gold",
                                              "green2", "green yellow", "coral1",
                                              "SteelBlue1", "cyan2", "lavender",
                                              "OliveDrab1"), k=2))        
        for i in range(9):
            for j in range(12):
                self.canvas.create_rectangle(152 + j * 32, 40 + i * 32, 184 + j * 32,
                             72 + i * 32, fill=farba1, width=0)
                farba1, farba2 = farba2, farba1
            farba1, farba2 = farba2, farba1

    def precitaj_subor(self, meno_suboru):
        with open(meno_suboru, encoding='utf-8') as subor:
            nazov = subor.readline()[7:]
            self.zoznam_sipiek = subor.readline()[8:].split()
            
            subor.readline()
            rad = []
            for riadok in subor:
                for prvok in riadok.split():
                    rad.append(prvok)
                self.mapa_levelu.append(rad)
                rad = []
        self.canvas.create_text(324, 358, text=nazov, fill="black", font="Verdana 16")
        mapa = []        
        for riadok in self.mapa_levelu:
            rad = []
            for prvok in riadok:
                if prvok in ["→", "←", "↓", "↑", "v", "<", ">", "^"]:
                    rad.append("_")
                else:
                    rad.append(prvok)
            mapa.append(rad)
        PohybujuciObjekt.mapa_pohybujucich_objektov = mapa
    
    def vykresli_level(self):
        self.inicializuj_sipky()
        farba = random.choice(("red2", "forest green", "olive drab",
                               "navy", "blue4", "gray25",
                              "DarkOrchid4", "dark green", "brown4",
                               "deep pink", "purple4", "dark slate gray"))
        steny = []
        objekt_pod_stenou = []
        for i in range(len(self.mapa_levelu)):
            for j in range(len(self.mapa_levelu[i])):               
                if self.mapa_levelu[i][j] == "1":
                    if i % 2 == 0:
                        steny.append(self.canvas.create_line(152 + (j // 2 * 32), 40 + (i // 2 * 32),
                                                152 + (j // 2 * 32) + 32, 40 + (i // 2 * 32),
                                                fill = farba, width = 3))
                    else:
                        steny.append(self.canvas.create_line(152 + (j // 2 * 32), 40 + (i // 2 * 32),
                                                152 + (j // 2 * 32), 40 + (i // 2 * 32) + 32,
                                                fill = farba, width = 3))
                        
                elif self.mapa_levelu[i][j] in ["→", "←", "↓", "↑"]:
                    smer = {"→" : "R", "←" : "L", "↓" : "D", "↑" : "U"}[self.mapa_levelu[i][j]]
                    self.mysi.append(PohybujuciObjekt(i, j, smer)) #mys
                    
                elif self.mapa_levelu[i][j] in ["v", "<", ">", "^"]:
                    smer = {">" : "R", "<" : "L", "v" : "D", "^" : "U"}[self.mapa_levelu[i][j]]
                    self.macky.append(PohybujuciObjekt(i, j, smer, False)) #macka

                elif self.mapa_levelu[i][j] == "S": #silo/domcek
                    self.domceky.append(Dom(i, j))

                elif self.mapa_levelu[i][j] == "X": #jama
                    y = int((i - 1) / 2) * 32 
                    x = int((j - 1) / 2) * 32
                    objekt_pod_stenou.append(self.canvas.create_image(168 + x, 55 + y, image = self.policka[0]))

                for stena in steny:
                    for objekt in objekt_pod_stenou:
                        self.canvas.tag_raise(stena, objekt)

    def restart_levelu(self):
        self.rychlost = 50
        self.start_hry = False
        self.faza_kruhu = 0
        self.canvas.delete(self.kruh)
        self.kruh = None
        self.zastav_hru = False
        
        for animovany_objekt in self.mysi:
            self.canvas.delete(animovany_objekt.objekt)
            del animovany_objekt            
        self.mysi = []
        for animovany_objekt in self.macky:
            self.canvas.delete(animovany_objekt.objekt)
            del animovany_objekt            
        self.macky = []
        
        for i in range(len(self.mapa_levelu)):
            for j in range(len(self.mapa_levelu[i])):
                if self.mapa_levelu[i][j] in ["→", "←", "↓", "↑"]:
                    smer = {"→" : "R", "←" : "L", "↓" : "D", "↑" : "U"}[self.mapa_levelu[i][j]]
                    self.mysi.append(PohybujuciObjekt(i, j, smer)) #mys
                    
                elif self.mapa_levelu[i][j] in ["v", "<", ">", "^"]:
                    smer = {">" : "R", "<" : "L", "v" : "D", "^" : "U"}[self.mapa_levelu[i][j]]
                    self.macky.append(PohybujuciObjekt(i, j, smer, False)) #macka

    def inicializuj_sipky(self):
        n = 0
        for sipka in self.zoznam_sipiek:
            if n >= 6:
                self.sipky_v_panely.append(Sipka(96, 120 + (n % 6) * 32 + (n % 6) * 8, sipka))
            else:
                self.sipky_v_panely.append(Sipka(56, 120 + n * 32 + (n % 6) * 8, sipka))
            n += 1

    def kolizia_s_mackou(self):
        for macka in self.macky:
            x1, y1 = tuple(self.canvas.coords(macka.objekt))
            y1 += 16
            for mys in self.mysi:
                x2, y2 = tuple(self.canvas.coords(mys.objekt))
                vzd = (((x1 - x2) ** 2) + ((y1 - y2) ** 2)) ** (1/2)
                if vzd < 12:
                    mys.zomrel = True
                    mys.faza_animacie = 0

    def chyba(self, id_objektu):
        self.zastav_hru = True
        x, y = tuple(self.canvas.coords(id_objektu))
        self.kruh = self.canvas.create_image(x, y, image=[self.kruhy[0]])

    def animuj_kruh(self):
        if self.faza_kruhu < 20:
            self.faza_kruhu += 1
            self.canvas.itemconfig(self.kruh, image = self.kruhy[self.faza_kruhu])
        else: #treba sem nieco dopisat?
            pass
                

class PohybujuciObjekt(Level):
    canvas = None
    mapa_pohybujucich_objektov = None
    chuchu_d = None
    chuchu_r = None
    chuchu_l = None
    chuchu_u = None
    chuchu_anj = None
    chuchu_pad = None
    kapukapu_d = None
    kapukapu_r = None
    kapukapu_l = None
    kapukapu_u = None
    kapukapu_pad = None
    
    def __init__(self, riadok, stlpec, smer, chuchu=True):
        self.stlpec = stlpec
        self.riadok = riadok
        self.smer = smer
        self.chuchu = chuchu
        self.faza_animacie = 0
        self.pocet_krokov = 0
        self.animacie = []
        self.objekt = None
        self.vymaz = False
        self.zomrel = False
        y = int((riadok - 1) / 2) * 32 
        x = int((stlpec - 1) / 2) * 32
        if self.chuchu:
            self.objekt = self.canvas.create_image(168 + x, 54 + y)
            self.pocet_krokov = 5
        else:
            self.objekt = self.canvas.create_image(168 + x, 38 + y)
            self.pocet_krokov = 8

    def interpretuj_policko(self):
            policko = self.mapa_pohybujucich_objektov[self.riadok][self.stlpec]
            if policko == "_":
                pass
            elif policko == "L":
                self.smer = "L"
            elif policko == "R":
                self.smer = "R"
            elif policko == "U":
                self.smer = "U"
            elif policko == "D":
                self.smer = "D"
            elif policko == "S":
                self.vymaz = True
                self.smer = "X"
            elif policko == "X":
                self.smer = "@"
                self.zomrel = True
                self.faza_animacie = 0
                if self.chuchu is False:
                    self.canvas.move(self.objekt, 0, 14)
                
            
    def posun(self, posun_v_zozname = False):
        if self.chuchu:
            if (self.pocet_krokov == 5 or self.pocet_krokov == 1):
                posunutie = 7
            else:
                posunutie = 6
        else:
      
            posunutie = 4

        if (self.chuchu and self.pocet_krokov == 5) or (self.chuchu is not True and self.pocet_krokov == 8):
            self.interpretuj_policko()
            if self.smer == "L":
                if (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "1"):
                    
                    self.smer = "X"

                elif (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "1"):
                    self.stlpec +=2
                    self.smer = "R"
                    
                elif (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "1"):
                    self.riadok += 2
                    self.smer = "D"
                    
                elif self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "1":
                    self.riadok -= 2
                    self.smer = "U"

                elif self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "0":
                    self.stlpec -=2
                    self.smer = "L"
                    
            elif self.smer == "R":
                if (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "1"):
                    
                    self.smer = "X"

                elif (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "1"):
                    self.stlpec -=2
                    self.smer = "L"
                    
                elif (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "1"):
                    self.riadok -= 2
                    self.smer = "U"
                    
                elif self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "1":
                    self.riadok += 2
                    self.smer = "D"

                elif self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "0":
                    self.stlpec +=2
                    self.smer = "R"

            elif self.smer == "U":
                if (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "1"):
                    
                    self.smer = "X"

                elif (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "1"):
                    self.riadok +=2
                    self.smer = "D"
                    
                elif (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "1"):
                    self.stlpec -=2
                    self.smer = "L"
                    
                elif self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "1":
                    self.stlpec +=2
                    self.smer = "R"

                elif self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "0":
                    self.riadok -=2
                    self.smer = "U"

            elif self.smer == "D":
                if (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok - 1][self.stlpec] == "1" and
                    self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "1"):
                    
                    self.smer = "X"

                elif (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok][self.stlpec + 1] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "1"):
                    self.riadok -=2
                    self.smer = "U"
                    
                elif (self.mapa_pohybujucich_objektov[self.riadok][self.stlpec - 1] == "1" and
                      self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "1"):
                    self.stlpec +=2
                    self.smer = "R"
                    
                elif self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "1":
                    self.stlpec -=2
                    self.smer = "L"

                elif self.mapa_pohybujucich_objektov[self.riadok + 1][self.stlpec] == "0":
                    self.riadok +=2
                    self.smer = "D"
            self.pocet_krokov = 0
        self.animuj()
        if self.smer != "X" and self.smer != "@":
            self.pocet_krokov += 1
        
        if self.smer == "L":
            self.canvas.move(self.objekt, -posunutie, 0)
        elif self.smer == "R":
            self.canvas.move(self.objekt, posunutie, 0)
        elif self.smer == "U":
            self.canvas.move(self.objekt, 0, -posunutie)
        elif self.smer == "D":
            self.canvas.move(self.objekt, 0, posunutie)

            
    def animuj(self):
        
        self.canvas.tag_raise(self.objekt)

        if self.zomrel and self.chuchu and self.smer !="@":
            self.animacie = self.chuchu_anj
            self.smer = "X"
            self.canvas.move(self.objekt, 0, -2)
            self.canvas.tag_raise(self.objekt)
            
        if self.smer == "L":
            if self.chuchu:
                self.animacie = self.chuchu_l
            else:
                self.animacie = self.kapukapu_l
        elif self.smer == "R":
            if self.chuchu:
                self.animacie = self.chuchu_r
            else:
                self.animacie = self.kapukapu_r
        elif self.smer == "U":
            if self.chuchu:
                self.animacie = self.chuchu_u
            else:
                self.animacie = self.kapukapu_u
        elif self.smer == "D":
            if self.chuchu:
                self.animacie = self.chuchu_d
            else:
                self.animacie = self.kapukapu_d
        elif self.smer == "@":
            if self.chuchu:
                self.animacie = self.chuchu_pad
            else:
                self.animacie = self.kapukapu_pad
                
        if self.animacie:
            if self.smer == "@" and (self.faza_animacie + 1 == len(self.animacie)):
                self.vymaz = True
            else:   
                self.canvas.itemconfig(self.objekt, image = self.animacie[self.faza_animacie])        
                self.faza_animacie = (self.faza_animacie + 1) % len(self.animacie)

class Sipka:    
    canvas = None
    sipky = None
        
    def __init__(self, x, y, smer):        
        self.x0 = x
        self.y0 = y
        self.x = x
        self.y = y
        self.smer = smer
        self.kliknutie_na_sipku = False
        self.stvorec = self.canvas.create_image(x, y, image=self.sipky[{"R" : 1, "D" : 3, "L" : 5, "U" : 7}[self.smer]])

    def reset(self):
        self.canvas.move(self.stvorec, self.x0 - self.x, self.y0 - self.y)
        self.x = self.x0
        self.y = self.y0 

class Dom:
    canvas = None
    domcek = None
    rakety_UL = None
    rakety_UR = None
    rakety_DL = None
    rakety_DR = None
    raketa_dym = None
    
    def __init__(self, riadok, stlpec):
        self.jama = False
        self.faza_animacie = 0
        self.stlpec = stlpec
        self.riadok = riadok
        self.raketa = None
        self.dym = []
        self.raketa_anim = []
        self.dx = 0
        self.dy = 0
        y = int((riadok - 1) / 2) * 32 
        x = int((stlpec - 1) / 2) * 32
        self.dom = self.canvas.create_image(32 + 88 + 32 + x + 16, 44 + y + 12, image=self.domcek[1])

    def vstup(self):
        self.canvas.itemconfig(self.dom, image=self.domcek[0])        

    def reset(self):
        self.canvas.itemconfig(self.dom, image=self.domcek[1])
        
    def odpal(self):
        if self.faza_animacie == 0:
            PohybujuciObjekt.mapa_pohybujucich_objektov[self.riadok][self.stlpec] = "X"
            if self.stlpec > 6 and self.riadok < 5:
                self.dx = 2
                self.dy = -2
                self.raketa_anim = self.rakety_UR
            elif self.stlpec <= 6 and self.riadok < 5:
                self.dx = -2
                self.dy = -2
                self.raketa_anim = self.rakety_UL
            if self.stlpec > 6 and self.riadok >= 5:
                self.dx = 2
                self.dy = 2
                self.raketa_anim = self.rakety_DR
            elif self.stlpec <= 6 and self.riadok >= 5:
                self.dx = -2
                self.dy = 2
                self.raketa_anim = self.rakety_DL
                
            y = int((self.riadok - 1) / 2) * 32 
            x = int((self.stlpec - 1) / 2) * 32
            self.canvas.itemconfig(self.dom, image=self.domcek[2])
            self.dym.append(self.canvas.create_image(32 + 88 + 32 + x + 16, 44 + y + 12,
                                                          image=self.raketa_dym[0]))
            self.raketa = self.canvas.create_image(32 + 88 + 32 + x + 16, 44 + y + 12,
                                                   image=self.raketa_anim[0])
            self.faza_animacie += 1
            self.jama = True
        else:
            self.animuj()

    def animuj(self):
        if self.faza_animacie < len(self.raketa_anim):
            y = int((self.riadok - 1) / 2) * 32 
            x = int((self.stlpec - 1) / 2) * 32
            self.dym.append(self.canvas.create_image(32 + 88 + 32 + x + 16 +
                                                     self.faza_animacie * self.dx,
                                                     44 + y + 12 + self.faza_animacie * self.dy,
                                                     image=self.raketa_dym[self.faza_animacie]))
            if len(self.dym) > 10:
                self.canvas.delete(self.dym[0])
                self.dym.pop(0)
            self.canvas.tag_raise(self.dym[-1])
            self.canvas.itemconfig(self.raketa, image=self.raketa_anim[self.faza_animacie])                    
            self.canvas.tag_raise(self.raketa)
            self.canvas.move(self.raketa, self.dx, self.dy)
            self.faza_animacie += 1
        else:
            [self.canvas.delete(self.dym[i]) for i in range(10)]
            self.canvas.delete(self.raketa)
                
Hra()
