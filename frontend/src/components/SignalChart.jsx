import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function SignalChart({ signalData }) {
    if (!signalData || signalData.length === 0) return null;

    const data = signalData.map((val, i) => ({ idx: i, amplitude: val }));

    return (
        <div className="chart-card glass-card">
            <div className="chart-title">📈 Vibration Signal Waveform</div>
            <ResponsiveContainer width="100%" height={280}>
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                        dataKey="idx"
                        tickFormatter={v => v}
                        label={{ value: 'Sample', position: 'insideBottomRight', offset: -5, style: { fill: '#5e5e80', fontSize: 12 } }}
                    />
                    <YAxis
                        label={{ value: 'Amplitude', angle: -90, position: 'insideLeft', style: { fill: '#5e5e80', fontSize: 12 } }}
                    />
                    <Tooltip
                        contentStyle={{ background: 'rgba(17,17,40,0.95)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }}
                        labelStyle={{ color: '#9a9abf' }}
                        itemStyle={{ color: '#667eea' }}
                    />
                    <Line type="monotone" dataKey="amplitude" stroke="#667eea" strokeWidth={1.5} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
