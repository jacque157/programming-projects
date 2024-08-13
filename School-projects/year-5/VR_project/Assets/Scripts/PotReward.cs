using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PotReward : MonoBehaviour
{
    // Start is called before the first frame update
    public Animator animator;
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void Reward()
    {
        animator.Play("openGate");
    }
}
