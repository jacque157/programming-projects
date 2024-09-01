using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Player
{
    public enum Weapon
    {
        revolver,
        spell
    }

    public static float maxHealth = 100;
    public static float health = maxHealth;  
    public static float maxSanity = 120;
    public static float sanity = maxSanity;

    public static float x, y, z;
    private static List<Weapon> inventory = new List<Weapon>()
    {
        Weapon.revolver,
        Weapon.spell,
    };
    private static int inventoryIndex = 0;

    public static int cylinderSize = 6;
    public static int bulletsInCylinder = cylinderSize;
    public static int revolverAmmo = 5;
    public static int maxRevolverAmmo = 15;
    public static float revolverDamage = 24f;

    public static float spellHorror = 36f;

    public static void Damage(float damageDealt)
    {
        if (damageDealt > health)   
            health = 0;    
        else
            health -= damageDealt;       
    }

    public static void HealHealth(float amount)
    {
        if (amount + health >= maxHealth)
            health = maxHealth;
        else
            health += amount;
    }

    public static void Traumatise(float horrorDealt)
    {
        if (horrorDealt > sanity)
            sanity = 0;
        else
            sanity -= horrorDealt;
    }

    public static void HealHorror(float amount)
    {
        if (amount + sanity >= maxSanity)
            sanity = maxSanity;
        else
            sanity += amount;
    }

    public static void ReplenishRevolverAmmo(int amount)
    {
        if (revolverAmmo + amount >= maxRevolverAmmo)
            revolverAmmo = maxRevolverAmmo;
        else
            revolverAmmo += amount;
    }

    public static Weapon GetEquipedWeapon()
    {
        return inventory[inventoryIndex];
    }

    public static void NextWeapon()
    {
        inventoryIndex = (inventoryIndex + 1) % inventory.Count;
    }

    public static void PrevWeapon()
    {
        inventoryIndex = (inventoryIndex - 1 >= 0)? inventoryIndex - 1 : inventory.Count - 1;
    }
}
