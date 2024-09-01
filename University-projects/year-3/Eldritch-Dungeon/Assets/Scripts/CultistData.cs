using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "Data", menuName = "ScriptableObject/CultistData", order = 1)]
public class CultistData : ScriptableObject
{
    public float health;
    public float damage;
    public float horror;
    public float speed;
}
