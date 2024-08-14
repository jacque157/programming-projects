using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PotCollider : MonoBehaviour
{
    public ParticleSystem correctParticle;  // Particle system for correct ingredient
    public ParticleSystem wrongParticle;    // Particle system for wrong ingredient
    public ParticleSystem finalParticle;    // Particle system for final glow

    private int steaksCount = 0;
    private int onionsCount = 0;
    private int carrotsCount = 0;
    public PotReward potReward;

    private bool allIngredientsThrown = false;

    private void OnTriggerEnter(Collider other)
    {
        Destroy(other.gameObject);
        if (allIngredientsThrown) return;  // If all ingredients are thrown, no need to check further

        if (other.CompareTag("Steak"))
        {
            if (steaksCount == 3)
            {
                InstantiateParticleSystem(wrongParticle, transform.position, Quaternion.identity);
                return;
            }
            steaksCount++;
        }
        else if (other.CompareTag("Onion"))
        {
            if (onionsCount == 3)
            {
                InstantiateParticleSystem(wrongParticle, transform.position, Quaternion.identity);
                return;
            }
            onionsCount++;
        }
        else if (other.CompareTag("Carrot"))
        {
            if (carrotsCount == 3)
            {
                InstantiateParticleSystem(wrongParticle, transform.position, Quaternion.identity);
                return;
            }
            carrotsCount++;
        }
        else
        {
            InstantiateParticleSystem(wrongParticle, transform.position, Quaternion.identity);
            return;  
        }

        CheckIngredient();
    }

    void CheckIngredient()
    {
        // Check if all ingredients are thrown
        if (steaksCount == 3 && onionsCount == 3 && carrotsCount == 3)
        {
            allIngredientsThrown = true;
            // Final glow
            ParticleSystem ps = Instantiate(finalParticle, transform.position, Quaternion.identity);
            ps.Play();
            if(potReward != null)
                potReward.Reward();
        }
        else
        {
            // Correct ingredient
            Debug.Log("maam");
            InstantiateParticleSystem(correctParticle, transform.position, Quaternion.identity);
        }
    }

    void InstantiateParticleSystem(ParticleSystem particleSystem, Vector3 position, Quaternion rotation)
    {
        ParticleSystem ps = Instantiate(particleSystem, position, rotation);
        ps.Play();
        Destroy(ps.gameObject, ps.main.duration);
    }
}
