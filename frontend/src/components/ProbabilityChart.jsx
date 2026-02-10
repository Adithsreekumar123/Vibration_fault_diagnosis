import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const COLORS = ['#38ef7d', '#f5576c', '#00f2fe', '#fee140'];

export default function ProbabilityChart({ probabilities }) {
    if (!probabilities) return null;

    const data = Object.entries(probabilities).map(([name, value], i) => ({
        name: name.replace(' Fault', ''),
        probability: +(value * 100).toFixed(1),
        color: COLORS[i],
    }));

    return (
        <div className="chart-card glass-card">
            <div className="chart-title">📊 Class Probabilities</div>
            <ResponsiveContainer width="100%" height={280}>
                <BarChart data={data} barSize={50}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 12 }} />
                    <Tooltip
                        contentStyle={{ background: 'rgba(17,17,40,0.95)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }}
                        labelStyle={{ color: '#9a9abf' }}
                        formatter={(value) => [`${value}%`, 'Probability']}
                    />
                    <Bar dataKey="probability" radius={[6, 6, 0, 0]}>
                        {data.map((entry, index) => (
                            <Cell key={index} fill={entry.color} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}
