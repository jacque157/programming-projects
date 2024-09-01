using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Orb : MonoBehaviour
{
    public Vector3 direction = new Vector3(0,0,0);
    private float speed = 10f;//0.25f;
    public bool hurtsPlayer = false;
    public float damage = 30f;
    public float horror = 10f;

    // Start is called before the first frame update
    void Start()
    {

    }

    void OnTriggerEnter(Collider collision)
    {
        Animator animator = GetComponent<Animator>();

        if (animator.GetCurrentAnimatorStateInfo(0).IsName("OrbPulsing"))
        {
            if (!hurtsPlayer && collision.gameObject.tag == "Enemy")
            {
                collision.gameObject.GetComponent<Enemy>().Damage(damage + horror);
                animator.SetTrigger("collision");
            }

            if (hurtsPlayer && collision.gameObject.tag == "Player")
            {
                Player.Damage(damage);
                Player.Traumatise(horror);
                animator.SetTrigger("collision");
            }
        }       
    }

    // Update is called once per frame
    void Update()
    {
        SphereCollider hitbox = GetComponent<SphereCollider>();
        Animator animator = GetComponent<Animator>();
        if (Physics.CheckSphere(hitbox.transform.position, hitbox.radius, LayerMask.GetMask("Wall")) || 
            Physics.CheckSphere(hitbox.transform.position, hitbox.radius, LayerMask.GetMask("Floor")) ||
            transform.position.y < 0 || transform.position.y > 1.6 )
        {
            animator.SetTrigger("collision");
        }

        if (animator.GetCurrentAnimatorStateInfo(0).IsName("OrbPulsing"))
        {
            GetComponent<Transform>().position += direction.normalized * speed * Time.deltaTime;
        }

        if (animator.GetCurrentAnimatorStateInfo(0).IsName("Destroy"))
        {
            Destroy(gameObject);
        }
    }

}
