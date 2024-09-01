using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    public CharacterController controller;
    public Transform floorCheck;
    public float minimalFloorDistance = 0.2f;
    public LayerMask floorMask;

    public float speed = 2;
    public float gravity = -10f * 10;

    bool isGrounded;
    Vector3 gravitationalVelocity;

    // Update is called once per frame
    void Update()
    {
        isGrounded = Physics.CheckSphere(floorCheck.position, minimalFloorDistance, floorMask);
        if (isGrounded && gravitationalVelocity.y < 0)
            gravitationalVelocity.y = -1f;

        gravitationalVelocity.y += gravity *  Time.deltaTime;
        float x = Input.GetAxis("Horizontal");
        float z = Input.GetAxis("Vertical");

        Vector3 movement = speed * (transform.right * x + transform.forward * z);
        movement += gravitationalVelocity * Time.deltaTime;
        //print(movement);
        controller.Move(movement * Time.deltaTime);

        Transform bodyTransform = transform.GetChild(0).gameObject.transform;
        Player.x = bodyTransform.position.x;
        Player.y = bodyTransform.position.y;
        Player.z = bodyTransform.position.z;
        //print(bodyTransform.position);
    }
}
