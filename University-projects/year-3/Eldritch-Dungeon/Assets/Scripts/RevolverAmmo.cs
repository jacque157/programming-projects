using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RevolverAmmo : Item
{
    public static int ammoCount = 5;

    override public void Use()
    {
        if (Player.revolverAmmo >= Player.maxRevolverAmmo)
            return;
        Player.ReplenishRevolverAmmo(ammoCount);
        Destroy(gameObject);
    }
}
