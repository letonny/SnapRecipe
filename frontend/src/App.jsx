import { useState, useRef } from 'react'
import axios from 'axios'

// icons
const CameraIcon = () => (
  <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
)
const ChefIcon = () => (
  <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
)

function App() {
  const [step, setStep] = useState(1) 
  const [preview, setPreview] = useState(null)
  const [ingredients, setIngredients] = useState([])
  const [loading, setLoading] = useState(false)
  const [recipe, setRecipe] = useState(null)
  
  const [difficulty, setDifficulty] = useState("Intermediate")
  const [cuisine, setCuisine] = useState("Any")

  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setPreview(URL.createObjectURL(file))
      detectIngredients(file)
    }
  }

  const detectIngredients = async (file) => {
    setLoading(true)
    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await axios.post("http://127.0.0.1:8000/detect", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      })
      setIngredients(res.data.ingredients)
      setStep(2)
    } catch (err) {
      console.error(err)
      alert("Error detecting ingredients. Is backend running?")
    } finally {
      setLoading(false)
    }
  }

  const generateRecipe = async () => {
    setLoading(true)
    try {
      const payload = { ingredients, difficulty, cuisine }
      const res = await axios.post("http://127.0.0.1:8000/generate-recipe", payload)
      setRecipe(res.data)
      setStep(3)
    } catch (err) {
      console.error(err)
      alert("Error creating recipe. Check your API Key.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-base-200 flex flex-col items-center p-6 font-sans">
      
      {/* HEADER */}
      <header className="w-full max-w-2xl flex justify-between items-center mb-8 p-4">
        <div className="flex items-center gap-2">
           <div className="bg-primary p-2 rounded-lg text-primary-content shadow-lg">📸</div>
           <h1 className="text-2xl font-extrabold tracking-tight">SnapRecipe</h1>
        </div>
        <div className="badge badge-secondary badge-outline p-3">AI Chef Assistant</div>
      </header>

      {/* MAIN CARD CONTAINER */}
      <main className="w-full max-w-md card bg-base-100 shadow-xl overflow-hidden border border-base-300">
        
        {/* STEP 1: HERO & UPLOAD */}
        {step === 1 && (
          <div className="card-body items-center text-center space-y-6">
            <div className="w-32 h-32 bg-base-200 rounded-full flex items-center justify-center mb-2 animate-pulse">
              <span className="text-6xl">🥗</span>
            </div>
            <h2 className="card-title text-3xl">Fridge to Feast</h2>
            <p className="text-base-content/70">Snap a photo of your open fridge, and let our AI chef craft a personalized recipe instantly.</p>
            
            <input type="file" accept="image/*" className="hidden" ref={fileInputRef} onChange={handleFileChange} />
            
            <button 
              onClick={() => fileInputRef.current.click()}
              disabled={loading}
              className="btn btn-primary btn-lg w-full rounded-2xl shadow-lg"
            >
              {loading ? <span className="loading loading-spinner"></span> : <><CameraIcon /> Take Photo</>}
            </button>
          </div>
        )}

        {/* STEP 2: REVIEW & CONFIGURE */}
        {step === 2 && (
          <div className="card-body p-8">
            <figure className="relative mb-6 group">
              <img src={preview} alt="Fridge" className="w-full h-56 object-cover rounded-2xl shadow-md" />
              <button onClick={() => setStep(1)} className="btn btn-xs btn-circle absolute top-2 right-2">✕</button>
            </figure>

            <div className="mb-6">
              <h3 className="text-xs font-bold uppercase tracking-wider mb-3 opacity-50">Detected Ingredients</h3>
              <div className="flex flex-wrap gap-2">
                {ingredients.length > 0 ? ingredients.map((ing, idx) => (
                  <div key={idx} className="badge badge-accent badge-lg">
                    {ing}
                  </div>
                )) : <span className="italic opacity-50">No clear ingredients found.</span>}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-8">
              <div className="form-control">
                <label className="label"><span className="label-text font-bold">Difficulty</span></label>
                <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)} className="select select-bordered w-full">
                  <option>Beginner</option>
                  <option>Intermediate</option>
                  <option>Gourmet</option>
                </select>
              </div>
              <div className="form-control">
                <label className="label"><span className="label-text font-bold">Cuisine</span></label>
                <select value={cuisine} onChange={(e) => setCuisine(e.target.value)} className="select select-bordered w-full">
                  <option>Any</option>
                  <option>Italian</option>
                  <option>Mexican</option>
                  <option>Asian</option>
                </select>
              </div>
            </div>

            <button onClick={generateRecipe} disabled={loading} className="btn btn-neutral w-full h-14 text-lg">
              {loading ? "Chef is Thinking..." : <><ChefIcon /> Generate Recipe</>}
            </button>
          </div>
        )}

        {/* STEP 3: RECIPE CARD */}
        {step === 3 && recipe && (
          <div className="card-body max-h-[80vh] overflow-y-auto p-8">
            <button onClick={() => setStep(1)} className="btn btn-ghost btn-sm self-start mb-4">← Start Over</button>
            
            <div className="mb-6">
              <div className="badge badge-warning font-bold mb-2">{recipe.difficulty}</div>
              <h2 className="text-3xl font-extrabold leading-tight">{recipe.title}</h2>
              <p className="mt-2 flex items-center gap-2 opacity-70">⏱ {recipe.cook_time}</p>
            </div>

            <div className="alert alert-success bg-opacity-20 mb-8">
               <div>
                 <h3 className="font-bold text-success-content">👨‍🍳 Chef's Note</h3>
                 <p className="text-sm italic opacity-80">"{recipe.chef_note}"</p>
               </div>
            </div>

            <div className="space-y-6">
              <h3 className="font-bold text-xl border-b pb-2">Instructions</h3>
              <ol className="list-decimal list-inside space-y-4">
                {recipe.steps.map((step, i) => (
                  <li key={i} className="pl-2">
                    <span className="font-medium">{step.instruction}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        )}
      </main>
      <footer className="mt-8 opacity-50 text-sm">SnapRecipe © 2025</footer>
    </div>
  )
}

export default App