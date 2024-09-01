using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpiderBeta : EnemyBeta
{
    /*public float x = 0;
    public float y = 0;
    public float z = 0;
    public float scale = 0.5f;
    public string resourcesPath = "Enemies/Sprites/";
    GameObject spiderObject;

    public Spider(float x, float y, float z)
    {
        this.x = x;
        this.y = y;
        this.z = z;
        float y1 = y + (scale / 2);
        spiderObject = GameObject.Instantiate(GameObject.Find("Spider"));
        spiderObject.transform.localScale = new Vector3(scale, scale, scale);
        spiderObject.transform.position = new Vector3(x, y1, z);
    }*/

    new private void SetValues()
    {
        health = data.health;
        damage = data.damage;
        horror = data.horror;
        speed = data.speed;
    }

    void Update()
    {
        
        Animator animator = transform.GetChild(0).GetComponent<Animator>();

        if (health > 0)
        {
            Direction direction = GetDirectionFromPath();
            if (direction == Direction.none)
                animator.SetTrigger("stop");
            else
                animator.SetTrigger("walk");

            if (OccupiesPlayersSpace() && !animator.GetCurrentAnimatorStateInfo(0).IsName("SpiderBite"))
                animator.SetTrigger("attack");


            UpdatePath();
            Move(direction);
        }
        
        if ( ! animator.GetCurrentAnimatorStateInfo(0).IsName("SpiderDeath"))
            animator.SetTrigger("die");

    }

}
