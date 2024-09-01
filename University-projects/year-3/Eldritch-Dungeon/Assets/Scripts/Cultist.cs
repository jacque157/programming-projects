using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Cultist : Enemy
{
    public Orb orbPrefab;

    public float orbSpeed = 0.5f;

    public CultistData data;

    override public void SetValues()
    {
        health = data.health;
        damage = data.damage;
        horror = data.horror;
        speed = data.speed;
        orbPrefab = Resources.Load<Orb>("Prefabs/Orb");
    }

    // Update is called once per frame
    void Update()
    {
        Animator animator = GetComponent<Animator>();
        AnimatorStateInfo state = animator.GetCurrentAnimatorStateInfo(0);

        if (health > 0)
        {
            UpdatePath();
            if (state.IsName("CultistCast"))
                return;

            if (LineOfSight() && !state.IsName("CultistCast"))
            {
                animator.SetTrigger("stop");
                animator.SetTrigger("cast");
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
        else if (!state.IsName("CultistDeath"))
            animator.SetTrigger("die");
    }

    bool LineOfSight()
    {
        Vector3 position = transform.position;
        Vector3 direction = (new Vector3(Player.x, Player.y, Player.z) - position).normalized;
        
        if (Physics.Raycast(position, direction, out RaycastHit target))
        {
            if (target.collider.gameObject.tag == "Player")       
                return true;
            
        }
        return false;
    }

    public void Attack()
    {
        Vector3 position = transform.position;
        Vector3 direction = (new Vector3(Player.x, Player.y, Player.z) - position).normalized;

        SpawnOrb(position + (direction * 0.05f), direction);
    }

    public void SpawnOrb(Vector3 position, Vector3 direction)
    {
        Orb orb = Instantiate(orbPrefab, position, Quaternion.identity);
        orb.direction = direction * orbSpeed;
        orb.hurtsPlayer = true;
    }
}
