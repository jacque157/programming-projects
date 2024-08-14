using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class ActivateTeleportation : MonoBehaviour
{
    public GameObject TeleportationRay;
    public InputActionProperty RightAnalogPosition;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        Vector2 values = RightAnalogPosition.action.ReadValue<Vector2>();
        if (Mathf.Abs(values.x) < Mathf.Abs(values.y) && values.y < 0)
            TeleportationRay.SetActive(true);
        else 
            TeleportationRay.SetActive(false);
    }
}
