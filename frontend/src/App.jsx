function App() {
  return (
    <div>
      <header>
        <h1>Antarctica Navigation AI</h1>
        <p>AI-powered decision support for Antarctic research-vessel navigation</p>
      </header>

      <main>
        <section>
          <h2>Navigation Map</h2>
          <div>
            <p>Antarctic map will appear here.</p>
          </div>
        </section>

        <section>
          <h2>Navigation Overview</h2>

          <div>
            <h3>Sea-Ice Concentration</h3>
            <p>-- %</p>
          </div>

          <div>
            <h3>Iceberg Risk</h3>
            <p>--</p>
          </div>

          <div>
            <h3>Route Risk</h3>
            <p>--</p>
          </div>

          <div>
            <h3>Estimated Fuel</h3>
            <p>--</p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;