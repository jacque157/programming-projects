using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Bandage : Item
{
    public static float healAmount = 25;

    override public void Use()
    {
        if (Player.health >= Player.maxHealth)
            return;
        Player.HealHealth(healAmount);
        Destroy(gameObject);
    }
}
