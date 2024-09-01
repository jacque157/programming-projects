using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Spider : Enemy
{

    public SpiderData data;

    override public void SetValues()
    {
        health = data.health;
        damage = data.damage;
        horror = data.horror;
        speed = data.speed;
    }

    public void Bite()
    {
        if (PlayerColision())
        {
            Player.Damage(damage);
            Player.Traumatise(horror);
        }
            
    }

    void Update()
    {

        Animator animator = GetComponent<Animator>();
        AnimatorStateInfo state = animator.GetCurrentAnimatorStateInfo(0);

        if (health > 0)
        {
            UpdatePath();
            if (state.IsName("SpiderBite"))
                return;

            if (OccupiesPlayersSpace() && ! state.IsName("SpiderBite") && PlayerColision())
            {
                animator.SetTrigger("stop");
                animator.SetTrigger("attack");
                return;
            }
                     
            Direction direction = GetDirectionFromPath();

            if (direction == Direction.none)  
                animator.SetTrigger("stop"); 
            else
                animator.SetTrigger("walk");

            if (WallCollision(direction))
                return;

            Move(direction);
        }
        else if (! state.IsName("SpiderDeath") )
            animator.SetTrigger("die");

    }
}
